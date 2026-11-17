"""Parse the client's ground-truth docx-export into per-call Bengali transcripts +
gold medical terms (from the doc's terms table). Writes data/gold.json."""
import re, json
from pathlib import Path

VS = Path(__file__).resolve().parents[2]
lines = (VS / "data" / "client_transcripts.txt").read_text(encoding="utf-8").split("\n")

calls, cur, mode = [], None, None
for ln in lines:
    m = re.match(r"Call (\d+): (.+)", ln.strip())
    if m:
        cur = {"call": int(m.group(1)), "patient": m.group(2), "bn": []}
        calls.append(cur); mode = None; continue
    s = ln.strip()
    if s == "Bengali Transcription": mode = "bn"; continue
    if s == "English Transcription": mode = "en"; continue
    if s.startswith("=== TABLE"): mode = None; continue
    if cur and mode == "bn" and s:
        cur["bn"].append(re.sub(r"^(এজেন্ট|রোগীর আত্মীয়|রোগী)\s*[:：]\s*", "", s))

for c in calls:
    c["bengali"] = " ".join(c["bn"]); del c["bn"]

# gold terms per call, from the doc's "Medical Terms Used" table
GOLD = {
    1: {"medicines": [], "diagnoses": [], "procedures": ["tooth extraction", "root canal", "dental crown"], "tests": [], "symptoms": []},
    2: {"medicines": [], "diagnoses": ["paralysis"], "procedures": ["physiotherapy"], "tests": [], "symptoms": ["joint pain"]},
    3: {"medicines": ["eye drops"], "diagnoses": [], "procedures": ["injection"], "tests": [], "symptoms": []},
}
for c in calls:
    c["gold_terms"] = GOLD[c["call"]]

(VS / "data" / "gold.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8")
for c in calls:
    print(f"Call {c['call']} ({c['patient']}): {len(c['bengali'])} chars, {len(c['bengali'].split())} words")
