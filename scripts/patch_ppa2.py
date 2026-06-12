from pathlib import Path

path = Path(__file__).resolve().parent.parent / "ppa.py"
text = path.read_text(encoding="utf-8")
text_new = text.replace('return {"prediction": result}', 'return {"prediction": result, "source": "local"}')
if text != text_new:
    path.write_text(text_new, encoding="utf-8")
    print("patched ppa.py")
else:
    print("no changes made")
