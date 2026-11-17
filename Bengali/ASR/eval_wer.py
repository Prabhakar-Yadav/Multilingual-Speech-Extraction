"""Real WER: our Whisper transcripts vs the client's ground-truth Bengali.
Each of our 4 files is matched to its best gold call (min WER). No torch (jiwer only)."""
import json, jiwer
from pathlib import Path

VS = Path(__file__).resolve().parents[2]
gold = json.loads((VS / "data" / "gold.json").read_text(encoding="utf-8"))
res = json.loads((Path(__file__).resolve().parent / "results.json").read_text(encoding="utf-8"))

def collapse(t):
    out = []
    for w in t.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

for field, label in [("base_whisper_medium", "BASE"), ("finetuned_vaani", "FINE-TUNED")]:
    print(f"\n===== {label} — WER vs ground truth =====")
    wers = []
    for r in res:
        hyp = collapse(r[field])
        best = min(gold, key=lambda g: jiwer.wer(g["bengali"], hyp))
        w = 100 * jiwer.wer(best["bengali"], hyp)
        wers.append(w)
        print(f"  {r['file']} -> Call {best['call']} ({best['patient']:15}): WER {w:5.1f}%")
    print(f"  AVG WER: {sum(wers)/len(wers):.1f}%")
