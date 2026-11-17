"""Run the 4 real client audio files' Bengali transcripts through the fine-tuned
Bengali extraction model, to see what terms come out of the actual (degraded) audio.
"""
import sys, json, torch
from pathlib import Path

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from gen_data_bengali import SYS

BASE = str(B.parent / "Hindi" / "models" / "base-1b-4bit")
ADAPTER = str(B / "outputs" / "checkpoint-180")
OUTAUDIO = B.parent / "outputs_audio"

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

for n in ["1", "2", "3", "4"]:
    p = OUTAUDIO / n / "RESULT.json"
    if not p.exists():
        print(f"File {n}: no RESULT.json"); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    bn = d.get("transcript_bengali", "")
    enc = tok.apply_chat_template([{"role": "system", "content": SYS},
                                   {"role": "user", "content": bn}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False)
    pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"\n===== FILE {n} =====")
    print("BENGALI transcript (first 120 chars):", bn[:120])
    print("MODEL TERMS:", pred[:300])
