"""Evaluate the Bengali extractor (from the checkpoint) on 200 fresh unseen samples,
with an error breakdown (typo / missed / extra / bad JSON).

    .\.venv-train\Scripts\python.exe Bengali\eval200.py
"""
import sys, json, difflib
from pathlib import Path

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gen_data_bengali import make_rows

BASE = str(B.parent / "Hindi" / "models" / "base-1b-4bit")
ADAPTER = str(B / "outputs" / "checkpoint-180")     # trained adapter from the checkpoint
FIELDS = ["medicines", "diagnoses", "procedures", "tests", "symptoms"]

rows = make_rows(200, seed=int(sys.argv[1]) if len(sys.argv) > 1 else 999)

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def sim(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


exact = typos = missed = extra = badjson = 0
cases = []
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
        badjson += 1; continue
    if pj == gj:
        exact += 1; continue
    for k in FIELDS:
        g = [x for x in gj.get(k, [])]
        p = [x for x in pj.get(k, []) if isinstance(x, str)]
        gs, ps = {x.lower() for x in g}, {x.lower() for x in p}
        og, op = [x for x in g if x.lower() not in ps], [x for x in p if x.lower() not in gs]
        for x in og:
            typos += any(sim(x, y) >= 0.8 for y in op)
            missed += not any(sim(x, y) >= 0.8 for y in op)
        for y in op:
            extra += not any(sim(x, y) >= 0.8 for x in og)
    cases.append({"i": i, "bengali": r["messages"][1]["content"], "pred": pj, "gold": gj})

n = len(rows)
Path(B / "test200_report.json").write_text(json.dumps(
    {"exact": exact, "of": n, "accuracy_pct": round(100 * exact / n, 1),
     "errors": {"typos": typos, "missed": missed, "extra": extra, "bad_json": badjson},
     "wrong_cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n===== BENGALI 200-SAMPLE TEST =====")
print(f"EXACT MATCH : {exact}/{n}  ({round(100*exact/n,1)}%)")
print(f"  typos {typos} | missed {missed} | extra {extra} | bad_json {badjson}")
