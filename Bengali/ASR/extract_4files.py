"""Run the fine-tuned Bengali transcripts of the 4 client files through our 95%
medical-term extractor (Llama-3.2-1B + Bengali LoRA). Collapses repetition loops first.
"""
import sys, json, re, torch
from pathlib import Path

BASR = Path(__file__).resolve().parent
VS = BASR.parents[1]
sys.path.insert(0, str(VS / "Bengali"))
from gen_data_bengali import SYS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = str(VS / "Hindi" / "models" / "base-1b-4bit")
ADAPTER = str(VS / "Bengali" / "outputs" / "checkpoint-180")

def collapse(text):                          # kill consecutive repeated words (হোলো হোলো হোলো -> হোলো)
    out = []
    for w in text.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

results = json.loads((BASR / "results.json").read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

for r in results:
    bn = collapse(r["finetuned_vaani"])
    enc = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": bn}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=220, do_sample=False)
    pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    r["medical_terms"] = pred
    print(f"===== {r['file']} =====\n{pred}\n", flush=True)

(BASR / "results_with_terms.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved -> results_with_terms.json", flush=True)
