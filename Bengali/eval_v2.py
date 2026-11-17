"""Evaluate the v2 extractor: (1) the 4 REAL client transcripts should now come back
mostly EMPTY (no hallucinated symptoms); (2) held-out val accuracy stays high."""
import sys, json, torch
from pathlib import Path

B = Path(__file__).resolve().parent
VS = B.parent
sys.path.insert(0, str(B))
from gen_data_bengali import SYS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = str(VS / "Hindi" / "models" / "base-1b-4bit")
ADAPTER = str(B / "models" / "bengali-lora-1b-v2")

def collapse(text):
    out = []
    for w in text.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def extract(text):
    enc = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": text}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("===== 4 REAL CLIENT FILES (v2 - should be empty/minimal) =====", flush=True)
res = json.loads((B / "ASR" / "results.json").read_text(encoding="utf-8"))
for r in res:
    print(f"{r['file']}: {extract(collapse(r['finetuned_vaani']))}", flush=True)

print("\n===== held-out val accuracy (positives + negatives) =====", flush=True)
val = [json.loads(l) for l in (B / "data" / "val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:80]
exact = negs = neg_ok = 0
for r in val:
    gold = r["messages"][2]["content"]
    pred = extract(r["messages"][1]["content"])
    try:
        ok = json.loads(pred) == json.loads(gold)
    except Exception:
        ok = False
    exact += ok
    if gold.count("[]") == 5:
        negs += 1; neg_ok += ok
print(f"exact-match: {exact}/{len(val)}  |  negatives correct: {neg_ok}/{negs}", flush=True)
