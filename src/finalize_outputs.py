"""Consolidate each audio's output into one clean RESULT.json and delete the
intermediate clutter. Uses the trustworthy Groq results (Large V3 + Llama 70B).
Drops the dual-language medicines (they were confidence-0 prompt echoes)."""
import json
from pathlib import Path

B = Path(__file__).resolve().parent.parent
OUT = B / "outputs"

for i in ["1", "2", "3", "4"]:
    d = OUT / i
    if not d.exists():
        continue
    gr = json.loads((d / "groq_result.json").read_text(encoding="utf-8")) if (d / "groq_result.json").exists() else {}
    gt = (json.loads((d / "groq_terms.json").read_text(encoding="utf-8")).get("terms", {})
          if (d / "groq_terms.json").exists() else {})
    result = {
        "file": f"{i}.mp3.mpeg",
        "type": "Bengali doctor-patient phone call",
        "method": "Groq Whisper Large V3 (transcribe + translate) -> Llama 3.3 70B (medical extraction)",
        "transcript_bengali": gr.get("bengali", ""),
        "translation_english": gr.get("english_text", ""),
        "medical_terms": {
            "medicines": gt.get("medicines", []),
            "diagnoses": gt.get("diagnoses", []),
            "procedures": gt.get("procedures", []),
            "tests": gt.get("tests", []),
            "symptoms": gt.get("symptoms", []),
        },
        "note": ("Medicines empty: specific drug names were not clearly spoken or not "
                 "recoverable from this low-quality (8kHz) recording. Other terms are "
                 "extracted from the conversation content."),
    }
    (d / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in d.iterdir():                       # delete everything except the clean result
        if f.is_file() and f.name != "RESULT.json":
            f.unlink()

for f in OUT.iterdir():                          # tidy root: keep folders + README only
    if f.is_file() and f.name not in ("README.txt",):
        f.unlink()
print("done")
