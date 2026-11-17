"""Collapse Whisper long-form repetition artifacts in a transcript, then write
it in the format translate.py expects. (Whisper repeats words/phrases on
low-quality audio; this de-loops them before translation.)

    python src/clean_transcript.py <in.json> <out.json>
"""
import sys
import json
import re
from pathlib import Path

inp = sys.argv[1]
out = sys.argv[2]
d = json.loads(Path(inp).read_text(encoding="utf-8"))
t = d.get("transcript", "")

t = re.sub(r"(.)\1{3,}", r"\1\1", t)          # 4+ repeated chars -> 2  (ললললল -> লল)
t = re.sub(r"(\S+?)(?:[-\s]\1){2,}", r"\1", t)  # X-X-X or "X X X" -> X

words, res = t.split(), []
for w in words:                                # drop consecutive duplicate words
    if res and res[-1] == w:
        continue
    res.append(w)
t = " ".join(res)

Path(out).write_text(json.dumps(
    {"detected_language": d.get("detected_language", "bn"), "transcript": t},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"cleaned transcript ({len(t)} chars):\n{t[:600]}")
