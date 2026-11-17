"""Local GPU QLoRA fine-tune of the BENGALI medical-term extractor (Llama-3.2-1B),
same proven recipe as the Hindi run. Reuses the already-downloaded 4-bit base.
Trains, then evaluates on held-out Bengali and prints predicted vs gold.

    .\.venv-train\Scripts\python.exe Bengali\finetune_local.py
"""
import json, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

B = Path(__file__).resolve().parent
TRAIN = str(B / "data" / "train.jsonl")
VAL = str(B / "data" / "val.jsonl")
BASE = str(B.parent / "Hindi" / "models" / "base-1b-4bit")   # reuse downloaded 4-bit base


def run(base, seq, tag):
    print(f"\n=== training BENGALI extractor on {tag} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0}, dtype=torch.bfloat16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    ds = load_dataset("json", data_files=TRAIN, split="train")
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
                remove_columns=ds.column_names)

    trainer = SFTTrainer(model=model, train_dataset=ds, processing_class=tok,
        args=SFTConfig(dataset_text_field="text", max_length=seq, packing=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=2,
            learning_rate=2e-4, warmup_steps=5, logging_steps=25, optim="paged_adamw_8bit",
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none", output_dir=str(B / "outputs"), seed=11,
            save_strategy="steps", save_steps=90, save_total_limit=1))
    trainer.train()
    out = B / "models" / f"bengali-lora-{tag}"
    model.save_pretrained(str(out)); tok.save_pretrained(str(out))
    print("saved adapter ->", out, flush=True)
    return model, tok


def evaluate(model, tok, n=8):
    model.eval()
    rows = [json.loads(l) for l in Path(VAL).read_text(encoding="utf-8").splitlines()][:n]
    print("\n===== EVAL (Bengali in -> predicted terms vs gold) =====", flush=True)
    exact = 0
    for r in rows:
        enc = tok.apply_chat_template(r["messages"][:2], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=200, do_sample=False)
        pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gold = r["messages"][2]["content"]
        try:
            exact += json.loads(pred) == json.loads(gold)
        except Exception:
            pass
        print("\nBENGALI:", r["messages"][1]["content"][:80], "...")
        print("PRED :", pred[:200])
        print("GOLD :", gold)
    print(f"\nexact-match: {exact}/{len(rows)}")


if __name__ == "__main__":
    model, tok = run(BASE, 1024, "1b")
    evaluate(model, tok)
    print("\nDONE", flush=True)
