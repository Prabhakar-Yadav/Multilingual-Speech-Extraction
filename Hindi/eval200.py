"""Test the fine-tuned Hindi model on 200 FRESH unseen examples and break down
every error (typo / missed term / extra term / invalid JSON).

    .\.venv-train\Scripts\python.exe Hindi\eval200.py
"""
import sys, json, difflib
from pathlib import Path

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gen_data_hindi import make_rows

BASE = str(B / "models" / "base-1b-4bit")
ADAPTER = str(B / "models" / "hindi-lora-1b")
FIELDS = ["medicines", "diagnoses", "procedures", "tests", "symptoms"]

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 999
rows = make_rows(200, seed=SEED)                     # fresh set, different seed from training
(B / "data" / "test200.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def sim(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


exact = 0
errors, typos, missed, extra, badjson = [], [], [], [], 0
for i, r in enumerate(rows, 1):
    enc = tok.apply_chat_template(r["messages"][:2], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False)
    pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    gold = r["messages"][2]["content"]
    try:
        pj, gj = json.loads(pred), json.loads(gold)
    except Exception:
        badjson += 1
        errors.append({"i": i, "kind": "invalid_json", "pred": pred, "gold": gold})
        continue
    if pj == gj:
        exact += 1
        continue
    # term-level diff
    detail = {}
    for k in FIELDS:
        g = [x for x in gj.get(k, [])]
        p = [x for x in pj.get(k, [])]
        gs, ps = {x.lower() for x in g}, {x.lower() for x in p}
        only_g = [x for x in g if x.lower() not in ps]
        only_p = [x for x in p if x.lower() not in gs]
        for x in only_g:
            near = [y for y in only_p if sim(x, y) >= 0.8]
            (typos if near else missed).append(f"{k}: '{near[0]}' vs '{x}'" if near else f"{k}: '{x}'")
        for y in only_p:
            if not any(sim(x, y) >= 0.8 for x in only_g):
                extra.append(f"{k}: '{y}'")
        if only_g or only_p:
            detail[k] = {"missing_from_pred": only_g, "extra_in_pred": only_p}
    errors.append({"i": i, "kind": "term_diff", "diff": detail,
                   "hindi": r["messages"][1]["content"], "pred": pj, "gold": gj})

n = len(rows)
report = {
    "test_size": n, "exact_match": exact, "wrong": n - exact,
    "accuracy_pct": round(100 * exact / n, 1),
    "error_breakdown": {"typos_drug/term_spelling": len(typos), "missed_terms": len(missed),
                        "extra_terms": len(extra), "invalid_json": badjson},
    "typo_examples": typos[:15], "missed_examples": missed[:15], "extra_examples": extra[:15],
    "wrong_cases": errors,
}
(B / "test200_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n===== 200-SAMPLE TEST =====")
print(f"EXACT MATCH : {exact}/{n}  ({report['accuracy_pct']}%)")
print(f"WRONG       : {n-exact}")
print(f"  - spelling typos (right term, wrong spelling): {len(typos)}")
print(f"  - missed terms (in gold, not predicted)      : {len(missed)}")
print(f"  - extra terms (predicted, not in gold)       : {len(extra)}")
print(f"  - invalid JSON output                        : {badjson}")
print("report -> Hindi/test200_report.json")
