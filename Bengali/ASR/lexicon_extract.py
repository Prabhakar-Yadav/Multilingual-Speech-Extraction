"""HYBRID medical-term extractor.

Insight: the medical terms are a CLOSED vocabulary, and we now have the real Bengali
surface forms for each. So a deterministic lexicon matcher:
  * cannot hallucinate  (it only emits terms whose Bengali form is literally present)
  * cannot miss a known term that IS present
which is exactly the two failure modes the LLM kept hitting.

The LLM stays useful for paraphrases, but the lexicon guarantees precision.
This script scores the lexicon matcher on the 3 real client calls (gold).
"""
import sys, json, re, unicodedata
from pathlib import Path

VS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VS / "Bengali"))
from gen_data_bengali import MED, DIAG, PROC, SYMP, TEST

VOCAB = {"medicines": MED, "diagnoses": DIAG, "procedures": PROC, "tests": TEST, "symptoms": SYMP}
FIELDS = list(VOCAB)


def norm(s):
    """Bengali has Unicode composition variants - normalize both sides or matches silently fail."""
    return unicodedata.normalize("NFC", s)


def extract(text):
    """Emit an English term iff any of its Bengali surface forms appears in the text."""
    t = norm(text)
    out = {f: [] for f in FIELDS}
    for field, d in VOCAB.items():
        for eng, surfaces in d.items():
            if any(norm(s) in t for s in surfaces):
                out[field].append(eng)
    return out


if __name__ == "__main__":
    gold = json.loads((VS / "data" / "gold.json").read_text(encoding="utf-8"))
    tp = fp = fn = 0
    for c in gold:
        pred = extract(c["bengali"])
        g = c["gold_terms"]
        print(f"\n===== Call {c['call']} ({c['patient']}) =====")
        print("GOLD:", json.dumps({k: v for k, v in g.items() if v}, ensure_ascii=False))
        print("PRED:", json.dumps({k: v for k, v in pred.items() if v}, ensure_ascii=False))
        for f in FIELDS:
            gs = {x.lower() for x in g.get(f, [])}
            ps = {x.lower() for x in pred.get(f, [])}
            tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"\n===== LEXICON on gold: precision {prec:.2f} | recall {rec:.2f} | F1 {f1:.2f} "
          f"(tp={tp} fp={fp} fn={fn}) =====")
