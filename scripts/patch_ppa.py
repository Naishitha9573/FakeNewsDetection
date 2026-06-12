from pathlib import Path

path = Path(__file__).resolve().parent.parent / "ppa.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    '    return {"prediction": result}\n',
    '    return {"prediction": result, "source": "local"}\n',
)
path.write_text(text, encoding="utf-8")
print("patched ppa.py")
