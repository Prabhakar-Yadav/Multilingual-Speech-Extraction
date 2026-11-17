"""
PHASE 1 — Fine-tune Llama-3.2-3B for medical-term extraction (Unsloth QLoRA).

WHERE TO RUN: a GPU machine — free Google Colab or Kaggle (T4 16GB is enough).
NOT on the on-prem CPU box. Training data is synthetic (no PHI), so cloud GPU is fine.

STEPS
  1. Open https://colab.research.google.com  ->  Runtime -> Change runtime type -> T4 GPU.
  2. Upload data/train.jsonl (from this repo).
  3. Paste this file into a cell (or run as cells split on the '# %%' markers) and Run all.
  4. At the end it produces a GGUF. Download it into:
        VaultStack/models/llama-3.2-3b-medical-gguf/
     Then point src/extract_local.py's GGUF path at the new file. Done — on-prem, fine-tuned.
"""

# %% install (Colab)
# %pip install unsloth

# %% load base model in 4-bit
from unsloth import FastLanguageModel
import torch

MAX_SEQ = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,        # QLoRA: base in 4-bit -> ~6-8GB VRAM
)

# %% add LoRA adapters (only these train)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42,
)

# %% load + format the synthetic dataset (chat format -> single text field)
from datasets import load_dataset

ds = load_dataset("json", data_files="train.jsonl", split="train")
def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}
ds = ds.map(fmt)

# %% train (QLoRA SFT)
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAX_SEQ,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=5, num_train_epochs=3, learning_rate=2e-4,
        logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=42, output_dir="outputs", report_to="none",
    ),
)

# Focus learning on the assistant's JSON answer (not the prompt):
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer.train()

# %% quick sanity check
FastLanguageModel.for_inference(model)
msgs = [{"role": "system", "content": "You are a clinical NLP extractor. Return STRICT JSON "
         "with keys medicines, diagnoses, procedures, tests, symptoms."},
        {"role": "user", "content": "The doctor prescribed Amoxicillin and advised a chest X-ray "
         "for the pneumonia. The patient still has fever and cough."}]
inp = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(input_ids=inp, max_new_tokens=256)[0]))

# %% export to GGUF q4_k_m for on-prem llama.cpp (CPU)
model.save_pretrained_gguf("llama-3.2-3b-medical-gguf", tokenizer, quantization_method="q4_k_m")
# -> download llama-3.2-3b-medical-gguf/*.gguf into VaultStack/models/llama-3.2-3b-medical-gguf/
