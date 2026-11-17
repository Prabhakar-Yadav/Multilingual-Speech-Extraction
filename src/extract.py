"""Phase C - Extract medical terms from the English text.

Default engine: d4data/biomedical-ner-all (non-gated biomedical NER, CPU).
Outputs the terms to *report as needs* - medicines, procedures/surgeries,
diagnoses, tests. Availability is decided downstream by the hospital.

MedGemma 4B is the optional upgrade (better, but gated + heavy on CPU). When
you have an HF token + accepted license, we can add a `--engine medgemma` path.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import time
from pathlib import Path

from transformers import pipeline

_LOCAL = Path(__file__).resolve().parent.parent / "models" / "biomedical-ner-all"
NER_MODEL = str(_LOCAL) if _LOCAL.exists() else "d4data/biomedical-ner-all"

# Map the model's fine-grained entity groups to the buckets the hospital wants.
CATEGORY = {
    "Medication": "medicines",
    "Therapeutic_procedure": "procedures",       # incl. surgeries
    "Diagnostic_procedure": "tests",
    "Lab_value": "tests",
    "Disease_disorder": "diagnoses",
    "Sign_symptom": "symptoms",
}


def extract(text: str):
    ner = pipeline("token-classification", model=NER_MODEL, aggregation_strategy="simple")
    spans = sorted(ner(text), key=lambda s: s["start"])

    # The transformers pipeline leaves WordPiece fragments (az / ##ith / ##romycin).
    # Merge consecutive spans of the same entity that touch (gap <= 1 char also
    # joins hyphenated terms like X-ray), then slice the ORIGINAL text so we get
    # the clean surface form instead of "##" tokens.
    merged = []
    for s in spans:
        grp = s["entity_group"]
        if merged and merged[-1]["grp"] == grp and s["start"] - merged[-1]["end"] <= 1:
            merged[-1]["end"] = s["end"]
        else:
            merged.append({"grp": grp, "start": s["start"], "end": s["end"]})

    buckets = {v: [] for v in set(CATEGORY.values())}
    for m in merged:
        bucket = CATEGORY.get(m["grp"])
        if not bucket:
            continue
        term = text[m["start"]:m["end"]].strip()
        if term and term.lower() not in {t.lower() for t in buckets[bucket]}:
            buckets[bucket].append(term)
    return buckets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/translation.json")
    ap.add_argument("--out", default="outputs/medical_terms.json")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    english = data["english_text"]

    t0 = time.time()
    buckets = extract(english)
    elapsed = round(time.time() - t0, 1)

    result = {"engine": NER_MODEL, "processing_sec": elapsed, "terms": buckets}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processing time : {elapsed}s\nSaved           : {args.out}\n--- MEDICAL TERMS ---")
    print(json.dumps(buckets, ensure_ascii=False, indent=2))
