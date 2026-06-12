
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

import re
import pandas as pd
import joblib
import numpy as np
try:
    from scipy.sparse import hstack, csr_matrix
except Exception:
    raise ImportError("scipy is required by train_model.py. Install it with: pip install scipy")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^0-9a-zA-Z₹%\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_features(series: pd.Series):
    """Return numeric features for each text: currency, number, month, year, weekday, percent"""
    currency_re = re.compile(r"(₹|\brs\b|\binr\b)", re.I)
    number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
    month_re = re.compile(r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t)?|oct(ober)?|nov(ember)?|dec(ember)?)", re.I)
    year_re = re.compile(r"\b(19|20)\d{2}\b")
    weekday_re = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
    percent_re = re.compile(r"%")

    features = []
    for s in series.astype(str):
        s_low = s.lower()
        f_currency = 1 if currency_re.search(s_low) else 0
        f_number = 1 if number_re.search(s_low) else 0
        f_month = 1 if month_re.search(s_low) else 0
        f_year = 1 if year_re.search(s_low) else 0
        f_weekday = 1 if weekday_re.search(s_low) else 0
        f_percent = 1 if percent_re.search(s_low) else 0
        features.append([f_currency, f_number, f_month, f_year, f_weekday, f_percent])
    return np.array(features, dtype=float)


# Load datasets
if not os.path.exists('matter'):
    raise FileNotFoundError("Directory 'matter' not found. Make sure 'matter/Fake.csv' and 'matter/True.csv' exist.")

fake_path = os.path.join('matter', 'Fake.csv')
true_path = os.path.join('matter', 'True.csv')
if not os.path.exists(fake_path) or not os.path.exists(true_path):
    raise FileNotFoundError("Required data files not found: matter/Fake.csv or matter/True.csv")

fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Assign labels (0 = Fake, 1 = Real)
fake["label"] = 0
true["label"] = 1

# Combine and shuffle
data = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

# Build content and clean
data["content"] = (data.get("title", "") + " " + data.get("text", ""))
data["content"] = data["content"].apply(clean_text)

# Features and labels
X = data["content"].astype(str)
y = data["label"]

# Vectorizer with n-grams and sensible min_df to reduce noise
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.75, min_df=3)
X_vec = vectorizer.fit_transform(X)

# Numeric/textual engineered features
num_feats = extract_features(X)
X_vec = hstack([X_vec, csr_matrix(num_feats)])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# Base classifier with class balancing
base_clf = LogisticRegression(max_iter=2000, class_weight="balanced")

# Calibrate probabilities for more reliable confidence scores
clf = CalibratedClassifierCV(base_clf, cv=5)

# Train on training set
clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(clf, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer saved successfully.")