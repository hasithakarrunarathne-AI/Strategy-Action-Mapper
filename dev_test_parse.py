# dev_test_parse.py
from docx import Document
from src.ingest import extract_strategic_objectives, extract_actions

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

from pathlib import Path

strategic_text = read_docx(Path("data") / "STRATEGIC-PLAN-2024.docx")
action_text    = read_docx(Path("data") / "ACTION-PLAN-2024.docx")

strategies = extract_strategic_objectives(strategic_text)
actions    = extract_actions(action_text)

print("Strategies:", len(strategies))
print("Actions:", len(actions))

print("\nSample Strategy:", strategies[0] if strategies else "None")
print("\nSample Action:", actions[0] if actions else "None")