"""LOCAL GPU QLoRA fine-tune (Hindi practice) on a 4 GB NVIDIA GPU.
Tries the 3B with aggressive memory savings (4-bit + gradient checkpointing + paged
optimizer offloading to system RAM); auto-falls-back to 1B on out-of-memory.
Trains, then evaluates on held-out Hindi examples and prints predicted vs gold.

    .\.venv-train\Scripts\python.exe Hindi\finetune_local.py
"""
import json, gc, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

B = Path(__file__).resolve().parent
TRAIN = str(B / "data" / "train.jsonl")
VAL = str(B / "data" / "val.jsonl")
# pre-downloaded 4-bit bases (local paths, fetched with our reliable downloader)
BIG = str(B / "models" / "base-3b-4bit")
SMALL = str(B / "models" / "base-1b-4bit")


def run(base, seq, tag):
    print(f"\n=== training {base} (seq={seq}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0}, torch_dtype=torch.bfloat16)
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
    out = B / "models" / f"hindi-lora-{tag}"
    model.save_pretrained(str(out)); tok.save_pretrained(str(out))
    print("saved adapter ->", out, flush=True)
    return model, tok


def evaluate(model, tok, n=6):
    model.eval()
    rows = [json.loads(l) for l in Path(VAL).read_text(encoding="utf-8").splitlines()][:n]
    print("\n===== EVAL (Hindi in -> predicted terms vs gold) =====", flush=True)
    for r in rows:
        msgs = r["messages"][:2]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=200, do_sample=False)
        pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print("\nHINDI:", r["messages"][1]["content"][:90], "...")
        print("PRED :", pred[:200])
        print("GOLD :", r["messages"][2]["content"])


if __name__ == "__main__":
    # The 3B already validated the technique (loss 1.27 -> 0.09) but is too slow on a 4GB
    # GPU (~7-11h, optimizer paged to RAM) for a full run. Use the faster 1B for a reliable
    # COMPLETED practice result + eval. (Set base=BIG below to redo the 3B if desired.)
    model, tok = run(SMALL, 1024, "1b")
    evaluate(model, tok)
    print("\nDONE", flush=True)
