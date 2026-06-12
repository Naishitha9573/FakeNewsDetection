from pathlib import Path

path = Path(__file__).resolve().parent.parent / "requirements.txt" / "requirements.txt"
text = path.read_text(encoding="utf-8")
lines = text.splitlines()
required = [
    "fastapi",
    "uvicorn",
    "google-generative-ai",
]
for r in required:
    if r not in lines:
        lines.append(r)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Updated {path}")
