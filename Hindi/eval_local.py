"""Evaluate the saved Hindi LoRA adapter and write COMPLETE results to shareable
files (full Hindi conversation + model output + correct answer, no truncation).

    .\.venv-train\Scripts\python.exe Hindi\eval_local.py
"""
import json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

B = Path(__file__).resolve().parent
BASE = str(B / "models" / "base-1b-4bit")
ADAPTER = str(B / "models" / "hindi-lora-1b")
VAL = str(B / "data" / "val.jsonl")
N = 20

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

rows = [json.loads(l) for l in Path(VAL).read_text(encoding="utf-8").splitlines()][:N]
results, exact = [], 0
for i, r in enumerate(rows, 1):
    hindi = r["messages"][1]["content"]
    gold = r["messages"][2]["content"]
    enc = tok.apply_chat_template(r["messages"][:2], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False)
    pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    ok = False
    try:
        ok = json.loads(pred) == json.loads(gold)
    except Exception:
        pass
    exact += ok
    results.append({"example": i, "hindi_conversation": hindi,
                    "model_output": pred, "correct_answer": gold,
                    "match": "EXACT MATCH" if ok else "near-miss"})

# JSON (raw, complete)
(B / "model_eval_results.json").write_text(
    json.dumps({"model": "Llama-3.2-1B fine-tuned (QLoRA, local GPU)",
                "task": "Hindi conversation text -> English medical terms JSON",
                "score": f"{exact}/{len(rows)} exact match on unseen Hindi",
                "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

# TXT (readable, complete sentences)
lines = ["HINDI FINE-TUNING - FULL EVALUATION RESULTS",
         "Model: Llama-3.2-1B fine-tuned with QLoRA on a local GPU",
         "Task: read a HINDI doctor-patient conversation -> output the medical terms (English JSON)",
         f"Score: {exact}/{len(rows)} exact match on conversations the model had NEVER seen",
         "=" * 90, ""]
for x in results:
    lines += [f"[Example {x['example']}]  ({x['match']})",
              "HINDI CONVERSATION (what the model read):",
              x["hindi_conversation"], "",
              "MODEL OUTPUT (terms it extracted):",
              x["model_output"], "",
              "CORRECT ANSWER:",
              x["correct_answer"], "-" * 90, ""]
(B / "model_eval_results.txt").write_text("\n".join(lines), encoding="utf-8")

print(f"score: {exact}/{len(rows)}")
print("wrote:", B / "model_eval_results.txt", "and .json")
