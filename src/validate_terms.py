"""Local, OFFLINE drug-name validator + stock check (no LLM, no API, no internet).
Compares each extracted medicine to the hospital formulary using a similarity score
and a threshold, then flags:
   VERIFIED      (exact match)        -> trust + it's in stock
   FLAGGED       (close, likely typo) -> show suggested correct name for human confirm
   UNRECOGNIZED  (no close match)     -> needs human review (possible hallucination)

Runs in milliseconds on the same on-prem PC. The AI extracts; THIS verifies.

    python src/validate_terms.py <terms.json | "Med1,Med2,...">
"""
import sys, json, csv
from pathlib import Path
from rapidfuzz import process, fuzz

B = Path(__file__).resolve().parent.parent
FORMULARY = B / "data" / "formulary.csv"
FLAG_THRESHOLD = 85          # >= this but < 100 => likely typo; < this => unrecognized


def load_formulary():
    out = []
    with open(FORMULARY, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip() and row[0].strip().lower() != "name":
                out.append(row[0].strip())
    return out


def check(name, formulary):
    match, score, _ = process.extractOne(name, formulary, scorer=fuzz.ratio)
    if score >= 99.5:
        return {"input": name, "status": "VERIFIED", "matched": match,
                "score": round(score, 1), "in_stock": True, "needs_review": False}
    if score >= FLAG_THRESHOLD:
        return {"input": name, "status": "FLAGGED (likely typo)", "suggestion": match,
                "score": round(score, 1), "in_stock": False, "needs_review": True}
    return {"input": name, "status": "UNRECOGNIZED", "closest": match,
            "score": round(score, 1), "in_stock": False, "needs_review": True}


def get_meds(arg):
    p = Path(arg)
    if p.exists() and p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        t = d.get("medical_terms") or d.get("terms") or d
        return t.get("medicines", [])
    return [m.strip() for m in arg.split(",") if m.strip()]


if __name__ == "__main__":
    formulary = load_formulary()
    meds = get_meds(sys.argv[1])
    results = [check(m, formulary) for m in meds]
    icon = {"VERIFIED": "[OK]", "FLAGGED (likely typo)": "[!!]", "UNRECOGNIZED": "[XX]"}
    print(f"checked {len(results)} medicine(s) against {len(formulary)}-item formulary "
          f"(threshold {FLAG_THRESHOLD}%):\n")
    for r in results:
        tag = icon[r["status"]]
        extra = (f"-> did you mean '{r['suggestion']}'? ({r['score']}%)" if "suggestion" in r
                 else f"(in stock)" if r["in_stock"]
                 else f"(closest '{r['closest']}' {r['score']}%)")
        print(f"  {tag} {r['input']:<16} {r['status']:<22} {extra}")
    flagged = [r for r in results if r["needs_review"]]
    print(f"\n{len(results)-len(flagged)} auto-verified, {len(flagged)} flagged for human review.")
    Path(B / "outputs").mkdir(exist_ok=True)
    Path(B / "outputs" / "validation_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
