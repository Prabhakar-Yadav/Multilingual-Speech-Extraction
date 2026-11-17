"""Retrain the Bengali medical-term extractor WITH negative examples (v2), so it
stops hallucinating symptoms on follow-up/fee/feedback calls. No datasets/TRL
(they segfault here) - plain Trainer + custom dataset, loss only on the JSON answer.

    .\.venv-train\Scripts\python.exe -u Bengali\finetune_v2.py
"""
import json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

B = Path(__file__).resolve().parent
BASE = str(B.parent / "Hindi" / "models" / "base-1b-4bit")
OUT = B / "models" / "bengali-lora-1b-v4"
MAXLEN = 512          # activations (not LoRA rank) are what eat VRAM -> trade seq-len for capacity

# SAFETY: the NVIDIA driver BSOD'd twice at 97% VRAM. Hard-cap the process so it
# raises a clean CUDA-OOM instead of taking the driver (and Windows) down.
torch.cuda.set_per_process_memory_fraction(0.80)   # ~3.3GB PyTorch + ~0.2GB context = under 3.5GB total

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
rows = [json.loads(l) for l in (B / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
def _toklen(r):
    return len(tok(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False)["input_ids"])
rows = [r for r in rows if _toklen(r) <= MAXLEN]     # DROP long ones (never truncate the JSON answer)
rows = rows[:600]
print(f"training on {len(rows)} examples (all <= {MAXLEN} tokens)", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        m = self.rows[i]["messages"]
        full = tok.apply_chat_template(m, tokenize=False)
        prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(full, truncation=True, max_length=MAXLEN, add_special_tokens=False)["input_ids"]
        plen = min(len(tok(prompt, add_special_tokens=False)["input_ids"]), len(ids))
        labels = [-100] * plen + ids[plen:]              # loss only on the assistant JSON
        return {"input_ids": ids, "labels": labels}

def collate(batch):
    mx = max(len(b["input_ids"]) for b in batch)
    ii, ll, am = [], [], []
    for b in batch:
        pad = mx - len(b["input_ids"])
        ii.append(b["input_ids"] + [tok.pad_token_id] * pad)
        ll.append(b["labels"] + [-100] * pad)
        am.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll), "attention_mask": torch.tensor(am)}

model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, dtype=torch.bfloat16)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0, bias="none",   # FULL capacity (v3 level)
    task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=str(B / "outputs_v4"), per_device_train_batch_size=1, gradient_accumulation_steps=8,
    num_train_epochs=2, learning_rate=2e-4, warmup_steps=15, logging_steps=25, bf16=True,
    optim="paged_adamw_8bit", gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="steps", save_steps=25, save_total_limit=1, report_to="none")
import glob as _glob                                   # auto-resume: survives reboots/session cuts
_cks = [c for c in _glob.glob(str(B / "outputs_v4" / "checkpoint-*"))
        if (Path(c) / "trainer_state.json").exists()]        # skip half-written checkpoints
_cks.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
_resume = _cks[-1] if _cks else None
print("resuming from:", _resume, flush=True)
Trainer(model=model, args=args, train_dataset=DS(rows), data_collator=collate).train(
    resume_from_checkpoint=_resume)

model.save_pretrained(str(OUT)); tok.save_pretrained(str(OUT))
print("saved ->", OUT, flush=True)
