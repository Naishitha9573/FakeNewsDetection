import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
from dotenv import load_dotenv
try:
    from google import genai
except Exception:
    genai = None
import json
import re

# ---------------- ENV ---------------- #
load_dotenv()
_api_key_preview = os.getenv("GEMINI_API_KEY")
if _api_key_preview:
    print("GEMINI_API_KEY loaded: yes")
else:
    print("GEMINI_API_KEY loaded: no")

# ---------------- APP ---------------- #
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ML ---------------- #
try:
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
except Exception as e:
    model = None
    vectorizer = None
    print("Warning: could not load model/vectorizer:", e)
ML_CONFIDENCE_THRESHOLD = 0.5
GEMINI_CONFIDENCE_THRESHOLD = 0.7
MIN_WORDS_FOR_CONFIDENT_CLASSIFICATION = 5

# Knowledge base (loaded from data/facts.csv if present). Keep empty here — load external facts below.
KNOWLEDGE_BASE = []

# Load external facts file (editable) from `data/facts.csv` if present.
import csv
facts_path = os.path.join("data", "facts.csv")
if os.path.exists(facts_path):
    try:
        with open(facts_path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                phrase = (row.get('phrase') or row.get('content') or '').strip()
                label = (row.get('label') or '').strip()
                conf = row.get('confidence') or row.get('conf') or ''
                reason = row.get('reason') or ''
                if not phrase or not label:
                    continue
                try:
                    conf_val = float(conf) if conf != '' else 0.95
                except Exception:
                    conf_val = 0.95
                # store normalized phrase for more robust matching
                def _normalize_for_facts(s: str) -> str:
                    if not isinstance(s, str):
                        return ''
                    # map common smart quotes to ascii, lower, remove punctuation except digits/letters/spaces
                    s2 = s.replace('\u2019', "'").replace('\u2018', "'")
                    s2 = s2.replace('\u201c', '"').replace('\u201d', '"')
                    s2 = s2.lower()
                    s2 = re.sub(r"[^0-9a-z\s]", ' ', s2)
                    s2 = re.sub(r"\s+", ' ', s2).strip()
                    return s2

                KNOWLEDGE_BASE.append((_normalize_for_facts(phrase), label, conf_val, reason))
        print(f"Loaded {len(KNOWLEDGE_BASE)} facts from {facts_path}")
    except Exception as e:
        print("Failed to load facts file:", e)

# ---------------- RULE-BASED CHECK ---------------- #
def rule_based_check(news: str):
    lower = news.lower()

    # Knowledge base exact/substring matches
    for phrase, label, conf, reason in KNOWLEDGE_BASE:
        if phrase in lower:
            return label, conf, reason

    # Quick fake indicators
    keywords = ["viral", "rumor", "allegedly", "unverified", "whatsapp forward"]
    if any(word in lower for word in keywords):
        return "Fake News", 0.75, "Detected as unverified or viral claim"

    # Specific rule for COVID-19
    if re.search(r"\b(covid|coronavirus)\b", lower, re.I) and re.search(r"\b(2019|2020)\b", lower):
        return "Real News", 0.95, "Detected reference to COVID-19 outbreak in 2019/2020"

    # Known real events
    real_phrases = [
        "joe biden elected president",
        "india mars mission mangalyaan",
        "water on mars",
        "paris climate agreement",
        "bitcoin all-time high",
        "who pandemic declaration",
        "tesla model 3",
        "nobel peace prize world food programme"
    ]
    for phrase in real_phrases:
        if phrase in lower:
            return "Real News", 0.9, f"Detected known real event: {phrase}"

    # Heuristic: economic / price reports are often factual (currency + price keywords)
    currency_pattern = r"(₹|\brs\b|\brs\.\b|\binr\b)"
    price_keywords = r"(price|increased|decreased|rise|fell|hike|cut|raised|increase|decrease)"
    month_pattern = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t)?|oct(ober)?|nov(ember)?|dec(ember)?)"
    year_pattern = r"\b(19|20)\d{2}\b"

    has_currency = re.search(currency_pattern, lower, re.I)
    has_price_kw = re.search(price_keywords, lower, re.I)
    has_number = re.search(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b", lower)
    has_month_or_year = re.search(month_pattern, lower, re.I) or re.search(year_pattern, lower)

    if (has_currency or has_number) and has_price_kw:
        conf = 0.85
        if has_month_or_year or '%' in lower:
            conf = 0.9
        return "Real News", conf, "Detected economic/price report (currency + price keywords)"

    # -------- DATE / DAY HEURISTICS -------- #
    # Detect simple temporal/day statements like "today is Monday", "tomorrow is Tuesday",
    # or explicit dates like "March 23, 2026" and treat them as factual (Real News) for UI purposes.
    weekdays = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    temporal_words = r"(today|tomorrow|yesterday|tonight|this\s+morning|this\s+evening)"

    # Example matches: "today is monday", "tomorrow is tuesday?"
    if re.search(rf"\b{temporal_words}\b.*\b{weekdays}\b", lower) or re.search(rf"\b{weekdays}\b.*\b{temporal_words}\b", lower):
        return "Real News", 0.9, "Detected simple temporal/day statement"

    # Explicit calendar dates: day + month (+ year)
    day_num = r"\b([0-3]?\d)(st|nd|rd|th)?\b"
    if re.search(day_num, lower) and re.search(month_pattern, lower):
        # if year present, bump confidence
        conf = 0.85
        if re.search(year_pattern, lower):
            conf = 0.9
        return "Real News", conf, "Detected explicit calendar date"

    return None


def _extract_features_for_predict(text: str):
    """Return feature vector (list) that matches training numeric features order.
    Order: currency, number, month, year, weekday, percent
    """
    s = (text or "").lower()
    currency_re = re.compile(r"(₹|\brs\b|\binr\b)", re.I)
    number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
    month_re = re.compile(r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t)?|oct(ober)?|nov(ember)?|dec(ember)?)", re.I)
    year_re = re.compile(r"\b(19|20)\d{2}\b")
    weekday_re = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
    percent_re = re.compile(r"%")

    return [
        1 if currency_re.search(s) else 0,
        1 if number_re.search(s) else 0,
        1 if month_re.search(s) else 0,
        1 if year_re.search(s) else 0,
        1 if weekday_re.search(s) else 0,
        1 if percent_re.search(s) else 0,
    ]

# ---------------- GEMINI FUNCTION ---------------- #
def classify_with_gemini(news: str):
    # If the GenAI client or API key is not available, return uncertain.
    if genai is None:
        return "Uncertain", 0.5, "Gemini client not available"

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Uncertain", 0.5, "API key not found"

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert fact-checker. Classify the following news article as 'Fake News', 'Real News', or 'Uncertain'.

Guidelines:
- 'Real News': Factual information, verifiable events, credible reports from known sources.
- 'Fake News': False claims, conspiracy theories, misinformation, exaggerated or fabricated stories.
- 'Uncertain': Ambiguous content, satire, opinion pieces, or insufficient information to verify.

Consider:
- Historical facts (e.g., COVID-19 in 2019 is real)
- Scientific discoveries
- Political events
- Economic data
- Known hoaxes or myths

Respond ONLY in valid JSON format:
{{
  "label": "Real News" or "Fake News" or "Uncertain",
  "confidence": 0.0 to 1.0,
  "reason": "Brief explanation"
}}

News article:
{news}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    # -------- CLEAN JSON -------- #
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]

    clean_text = "\n".join(lines).strip()

    try:
        result = json.loads(clean_text)
        return (
            result.get("label", "Uncertain"),
            float(result.get("confidence", 0.5)),
            result.get("reason", "")
        )
    except:
        return "Uncertain", 0.5, text

# ---------------- ROUTES ---------------- #
@app.get("/")
def home():
    return {"message": "Fake News Detection API (Hybrid ML + Gemini + Rules)"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    news = data.get("news", "").strip()

    if not news:
        return {"error": "News text is empty."}

    # Very short snippets are usually not enough for a reliable fact-check verdict.
    if len(news.split()) < MIN_WORDS_FOR_CONFIDENT_CLASSIFICATION:
        return {
            "prediction": "Uncertain",
            "confidence": 0.4,
            "reason": "Input too short; provide a fuller claim with source/date/context.",
            "source": "validation"
        }

    # -------- RULE-BASED -------- #
    rule_result = rule_based_check(news)
    if rule_result:
        label, confidence, reason = rule_result
        return {
            "prediction": label,
            "confidence": confidence,
            "reason": reason,
            "source": "rule-based"
        }

    # -------- GEMINI -------- #
    try:
        gem_label, gem_conf, reason = classify_with_gemini(news)

        # Use Gemini only when it provides a strong confidence signal.
        if gem_label in ["Real News", "Fake News"] and float(gem_conf) >= GEMINI_CONFIDENCE_THRESHOLD:
            return {
                "prediction": gem_label,
                "confidence": round(gem_conf, 2),
                "reason": reason,
                "source": "gemini"
            }

        # -------- FALLBACK TO ML -------- #
        # If model or vectorizer were not loaded, return Uncertain
        if model is None or vectorizer is None:
            return {
                "prediction": "Uncertain",
                "confidence": 0.5,
                "reason": "ML model not available",
                "source": "ml"
            }

        # Build feature vector consistent with training: TF-IDF + engineered numeric features
        tf_vec = vectorizer.transform([news.lower()])
        try:
            from scipy.sparse import hstack, csr_matrix
        except Exception:
            return {
                "prediction": "Uncertain",
                "confidence": 0.5,
                "reason": "scipy is required for prediction but not available",
                "source": "ml"
            }

        num_feats = csr_matrix([_extract_features_for_predict(news)])
        vec = hstack([tf_vec, num_feats])

        probs = model.predict_proba(vec)[0]
        ml_conf = float(max(probs))

        ml_pred_num = model.predict(vec)[0]
        ml_label = "Real News" if ml_pred_num == 1 else "Fake News"

        reason_text = "Fallback to ML due to uncertainty in Gemini"
        if ml_conf < ML_CONFIDENCE_THRESHOLD:
            return {
                "prediction": "Uncertain",
                "confidence": round(ml_conf, 2),
                "reason": f"Low ML confidence ({ml_conf:.2f}); unable to classify reliably.",
                "source": "hybrid"
            }

        return {
            "prediction": ml_label,
            "confidence": round(ml_conf, 2),
            "reason": reason_text,
            "source": "hybrid"
        }

    except Exception as e:
        # On unexpected error, try to use ML if available. Otherwise return Uncertain.
        try:
            if model is not None and vectorizer is not None:
                tf_vec = vectorizer.transform([news.lower()])
                from scipy.sparse import hstack, csr_matrix
                num_feats = csr_matrix([_extract_features_for_predict(news)])
                vec = hstack([tf_vec, num_feats])
                probs = model.predict_proba(vec)[0]
                ml_conf = float(max(probs))
                ml_pred_num = model.predict(vec)[0]
                ml_label = "Real News" if ml_pred_num == 1 else "Fake News"
                if ml_conf < ML_CONFIDENCE_THRESHOLD:
                    return {
                        "prediction": "Uncertain",
                        "confidence": round(ml_conf, 2),
                        "reason": f"Recovered from error but ML confidence is low; {str(e)}",
                        "source": "ml"
                    }
                return {
                    "prediction": ml_label,
                    "confidence": round(ml_conf, 2),
                    "reason": f"Recovered from error; {str(e)}",
                    "source": "ml"
                }
        except Exception:
            pass

        return {
            "prediction": "Uncertain",
            "confidence": 0.5,
            "reason": f"System error: {str(e)}",
            "source": "error"
        }


if __name__ == "__main__":
    # Safe local run for development
    try:
        import uvicorn
        uvicorn.run("ppa:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print("Failed to start uvicorn from __main__:", e)