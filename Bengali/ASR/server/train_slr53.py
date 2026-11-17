"""
Bengali ASR fine-tune on the shared GPU server (RTX 5090, 24GB).

Dataset : OpenSLR SLR53 - clean READ Bengali (this is the key; our old Vaani data
          was spontaneous conversation, where even papers only reach 24-31% WER).
Model   : whisper-large-v3 + LoRA.
Honest  : the eval set is held out BEFORE training and never trained on.

Run:  cd ~/bengali_asr && ./venv/bin/python train_slr53.py
Resume after any interruption: run the exact same command (auto-resumes).
"""
import os, glob, random, pickle
from pathlib import Path

import numpy as np
import torch
import librosa
import jiwer
from scipy import signal
from dataclasses import dataclass
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model

# ---------------- config ----------------
HOME     = Path.home() / "bengali_asr"
DATA     = HOME / "slr53"
RUN      = os.environ.get("RUN", "run2")     # separate dir -> never clobbers a previous run
CKPT     = HOME / f"ckpt_{RUN}"
MODEL    = "openai/whisper-large-v3"
N_PARTS  = int(os.environ.get("N_PARTS", 6))     # SLR53 has 10 parts (~19k clips each)
EPOCHS   = int(os.environ.get("EPOCHS", 4))
BATCH    = int(os.environ.get("BATCH", 12))      # sized to stay inside the 16GB cap
GRAD_ACC = int(os.environ.get("GRAD_ACC", 2))    # effective batch = 24
# LR 1e-3 collapsed the model (WER 41.8 -> 77.7 while loss still fell = degenerate
# generation / exposure bias). 1e-4 is the standard, stable choice for Whisper LoRA.
LR       = float(os.environ.get("LR", 1e-4))
N_EVAL   = 500
# checkpoint cadence: a pause costs at most this many steps of lost work
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", 300))
# fraction of training clips degraded to 8kHz telephone quality (0 = off).
# Set TELE_PROB=0 to reproduce the original clean-only run.
TELE_PROB = float(os.environ.get("TELE_PROB", 0.5))
MAX_SEC  = 20
CKPT.mkdir(parents=True, exist_ok=True)

# SHARED SERVER: hard-cap our VRAM to ~14GB of the 24GB so other users always
# have ~10GB free. If we ever exceed it we get a clean CUDA-OOM (our job fails),
# never a crash that takes down someone else's training.
VRAM_GB = float(os.environ.get("VRAM_GB", 16))
_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
torch.cuda.set_per_process_memory_fraction(min(VRAM_GB / _total_gb, 0.95))
print(f"VRAM cap: {VRAM_GB:.0f} GB of {_total_gb:.0f} GB "
      f"(leaving ~{_total_gb - VRAM_GB:.0f} GB for other users)", flush=True)

print(f"model={MODEL} parts={N_PARTS} epochs={EPOCHS} batch={BATCH}x{GRAD_ACC}", flush=True)

# ---------------- data ----------------
utt2text = {}
with open(HOME / "utt_spk_text.tsv", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            utt2text[p[0]] = p[2].strip()

files = glob.glob(str(DATA / "**" / "*.flac"), recursive=True)
pairs = []
for fp in files:
    t = utt2text.get(os.path.basename(fp).replace(".flac", ""))
    if t:
        pairs.append({"path": fp, "text": t})

random.seed(0)
random.shuffle(pairs)
eval_pairs, train_pairs = pairs[:N_EVAL], pairs[N_EVAL:]
print(f"TRAIN {len(train_pairs)} | HELD-OUT {len(eval_pairs)} (never trained on)", flush=True)
print(f"TELEPHONE AUG: {TELE_PROB:.0%} of training clips degraded to 8kHz "
      f"({'ON' if TELE_PROB > 0 else 'OFF'}); eval set stays clean", flush=True)

processor = WhisperProcessor.from_pretrained(MODEL, language="bengali", task="transcribe")


def telephone_degrade(wav, sr=16000):
    """Make clean 16kHz studio speech look like the client's phone calls.

    Measured on the client audio: 8 kHz sample rate, 90% of energy below
    650-1270 Hz, no clipping. SLR53 is clean READ speech at 16 kHz, so the model
    currently trains on a signal it never meets in production - that mismatch is
    the single biggest gap between 33.7% WER on SLR53 and 92% on real calls.

    Downsample -> band-limit -> upsample reproduces the information loss exactly:
    everything above 4 kHz is gone for good, which is what the phone channel does.
    """
    # 300-3400 Hz is the classic telephone passband
    sos = signal.butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    w = signal.sosfilt(sos, wav)
    # round-trip through 8 kHz: the high band is destroyed, not merely attenuated
    w = librosa.resample(w, orig_sr=sr, target_sr=8000)
    w = librosa.resample(w, orig_sr=8000, target_sr=sr)
    # match the client's quiet level (rms ~0.045) instead of studio loudness
    rms = np.sqrt((w ** 2).mean()) + 1e-9
    return (w * (random.uniform(0.03, 0.07) / rms)).astype(np.float32)


class SLR53(torch.utils.data.Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        wav, _ = librosa.load(p["path"], sr=16000)
        wav = wav[: MAX_SEC * 16000].astype(np.float32)
        # Degrade only a fraction of samples: the model must stay good on clean
        # audio too, and mixing conditions is what makes it robust to both.
        if self.augment and random.random() < TELE_PROB:
            wav = telephone_degrade(wav)
        return {"input_features": processor.feature_extractor(wav, sampling_rate=16000).input_features[0],
                "labels": processor.tokenizer(p["text"]).input_ids[:448]}


# Held-out eval stays CLEAN so its WER remains comparable to the 33.71% baseline.
# Real telephone performance is measured separately by eval_client.py on the
# actual client calls - that is the number that decides whether this worked.
train_ds = SLR53(train_pairs, augment=TELE_PROB > 0)
eval_ds = SLR53(eval_pairs, augment=False)


@dataclass
class Collator:
    processor: object

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        # model weights are bf16; feature extractor gives float32. Without this cast,
        # eval/generate crashes: "Input type (float) and bias type (BFloat16) should be the same"
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        labs = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        batch["labels"] = labs["input_ids"].masked_fill(labs.attention_mask.ne(1), -100)
        return batch


def compute_metrics(pred):
    pid, lid = pred.predictions, pred.label_ids
    lid = np.where(lid == -100, processor.tokenizer.pad_token_id, lid)
    hyp = processor.batch_decode(pid, skip_special_tokens=True)
    ref = processor.batch_decode(lid, skip_special_tokens=True)
    pr = [(r, h) for r, h in zip(ref, hyp) if r.strip()]
    if not pr:
        return {"wer": 100.0, "cer": 100.0}
    r, h = zip(*pr)
    return {"wer": 100 * jiwer.wer(list(r), list(h)),
            "cer": 100 * jiwer.cer(list(r), list(h))}


# ---------------- model ----------------
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
model.generation_config.language = "bengali"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()

args = Seq2SeqTrainingArguments(
    output_dir=str(CKPT),
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    warmup_steps=200,
    num_train_epochs=EPOCHS,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    per_device_eval_batch_size=8,
    # Save every SAVE_STEPS steps and KEEP THEM ALL (save_total_limit=None).
    # Epoch-only saving meant a mid-epoch stop threw away all the work since the
    # last epoch boundary - we lost ~250 steps that way. Step-based saving means
    # a pause at any moment costs at most SAVE_STEPS of progress.
    # (An earlier run used save_total_limit=2, which deleted the best checkpoint.)
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=None,          # <-- keep every epoch's checkpoint
    load_best_model_at_end=True,    # <-- final saved model = the BEST WER, not the last
    metric_for_best_model="wer",
    greater_is_better=False,        # WER: lower is better
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=4,
)

# auto-resume from the newest VALID checkpoint
cks = [c for c in glob.glob(str(CKPT / "checkpoint-*"))
       if os.path.exists(os.path.join(c, "trainer_state.json"))]
cks.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
resume = cks[-1] if cks else None
print("resuming from:", resume if resume else "scratch", flush=True)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=Collator(processor),
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)
trainer.train(resume_from_checkpoint=resume)

# ---- per-epoch summary so the best epoch is obvious at a glance ----
print("\n" + "=" * 58, flush=True)
print("  EPOCH  |    WER   |    CER   |  word acc | char acc", flush=True)
print("=" * 58, flush=True)
best = None
for h in trainer.state.log_history:
    if "eval_wer" in h:
        w, c, e = h["eval_wer"], h["eval_cer"], h.get("epoch", 0)
        print(f"   {e:>4.0f}   |  {w:6.2f}% |  {c:6.2f}% |   {100-w:5.1f}%  |   {100-c:5.1f}%", flush=True)
        if best is None or w < best[1]:
            best = (e, w, c)
if best:
    print("=" * 58, flush=True)
    print(f"  BEST = epoch {best[0]:.0f}: WER {best[1]:.2f}% (word {100-best[1]:.1f}%) | "
          f"CER {best[2]:.2f}% (char {100-best[2]:.1f}%)", flush=True)
    print("  ^ load_best_model_at_end=True -> THIS is what gets saved", flush=True)

out = HOME / f"bengali_asr_final_{RUN}"
model.save_pretrained(str(out))
processor.save_pretrained(str(out))
print("saved ->", out, flush=True)

# keep the exact held-out split so the final eval is provably honest
with open(HOME / "eval_pairs.pkl", "wb") as f:
    pickle.dump(eval_pairs, f)
print("DONE", flush=True)
