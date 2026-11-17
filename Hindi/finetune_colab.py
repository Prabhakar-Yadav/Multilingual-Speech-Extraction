"""
HINDI PRACTICE — Fine-tune Llama-3.2-3B for Hindi medical-term extraction (Unsloth QLoRA).
Same technique as the main (Bengali/English) Phase-1 fine-tune — repeated in Hindi to
validate the workflow. Run on a GPU (free Google Colab T4).

STEPS
  1. Colab -> Runtime -> Change runtime type -> T4 GPU.
  2. Upload Hindi/data/train.jsonl (the Hindi dataset).
  3. First cell: !pip install unsloth   (then run the rest).
  4. Download the GGUF at the end -> Hindi/models/hindi-medical.gguf.

NOTE on base model: Llama-3.2-3B has decent Hindi; if Hindi quality is weak, swap
model_name to "unsloth/Qwen2.5-3B-Instruct" (stronger multilingual) — everything else stays the same.
"""
# %%
# !pip install unsloth

# %%
from unsloth import FastLanguageModel
MAX_SEQ = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=MAX_SEQ, load_in_4bit=True)

# %%
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=11)

# %%
from datasets import load_dataset
ds = load_dataset("json", data_files="train.jsonl", split="train")
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(
    ex["messages"], tokenize=False, add_generation_prompt=False)})

# %%
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(dataset_text_field="text", max_seq_length=MAX_SEQ,
        per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=5,
        num_train_epochs=3, learning_rate=2e-4, logging_steps=10, optim="adamw_8bit",
        weight_decay=0.01, lr_scheduler_type="linear", seed=11, output_dir="outputs", report_to="none"))
trainer = train_on_responses_only(trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")
trainer.train()

# %% quick check (Hindi in -> English terms out)
FastLanguageModel.for_inference(model)
msgs = [{"role": "system", "content": "Extract medical terms as STRICT JSON with keys "
         "medicines, diagnoses, procedures, tests, symptoms (English)."},
        {"role": "user", "content": "डॉक्टर ने पैरासिटामोल दिया है, मरीज़ को बुखार और खांसी है, "
         "और एक्स-रे करवाने को कहा है।"}]
inp = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(input_ids=inp, max_new_tokens=256)[0]))

# %% export GGUF for on-prem llama.cpp
model.save_pretrained_gguf("hindi-medical-gguf", tokenizer, quantization_method="q4_k_m")
# -> download the .gguf into VaultStack/Hindi/models/hindi-medical.gguf
