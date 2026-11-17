"""Test the v2 extractor on the CLEAN gold Bengali transcripts vs gold terms.
Isolates extraction quality from ASR error."""
import sys, json, torch
from pathlib import Path

VS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VS / "Bengali"))
from gen_data_bengali import SYS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = str(VS / "Hindi" / "models" / "base-1b-4bit")
ADAPTER = str(VS / "Bengali" / "models" / "bengali-lora-1b-v4")   # v4: multi-surface vocab, fully trained
gold = json.loads((VS / "data" / "gold.json").read_text(encoding="utf-8"))

tok = AutoTokenizer.from_pretrained(BASE)          # checkpoint has no tokenizer; use base
model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def extract(text):
    enc = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": text}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

FIELDS = ["medicines", "diagnoses", "procedures", "tests", "symptoms"]
tp = fp = fn = 0
for c in gold:
    pred = extract(c["bengali"])
    print(f"\n===== Call {c['call']} ({c['patient']}) =====", flush=True)
    print("GOLD:", json.dumps(c["gold_terms"], ensure_ascii=False))
    print("PRED:", pred)
    try:
        pj = json.loads(pred)
        for f in FIELDS:
            g = set(x.lower() for x in c["gold_terms"].get(f, []))
            p = set(x.lower() for x in pj.get(f, []) if isinstance(x, str))
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
    except Exception:
        pass

prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
print(f"\n===== EXTRACTION on CLEAN gold: precision {prec:.2f} | recall {rec:.2f} | F1 {f1:.2f} "
      f"(tp={tp} fp={fp} fn={fn}) =====", flush=True)
