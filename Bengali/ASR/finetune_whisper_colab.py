"""
BENGALI ASR FINE-TUNE on Vaani (with noise augmentation to match the client's
degraded phone audio). Fine-tunes Whisper on real field-recorded Bengali speech.

WHERE: Google Colab, T4 GPU.
RUN ORDER: run cells top-to-bottom (Runtime -> Run all). If the kernel ever
restarts (see app.log), ALL variables are wiped -> just Run all again.

This version is memory-safe: it converts each clip to features on the fly and
never holds all the raw audio in RAM at once (that OOM-restarted the kernel before).
"""

# %% 1. install
# !pip install -q transformers datasets peft accelerate bitsandbytes jiwer librosa soundfile scipy

# %% 2. login (token needs Vaani gated access)
from huggingface_hub import login
login(token="PASTE_YOUR_HF_TOKEN")

# %% 3. config + processor + augmentation
import random, gc, numpy as np, torch, librosa
from scipy.signal import butter, lfilter
from transformers import WhisperProcessor

BASE = "openai/whisper-small"          # bump to "openai/whisper-medium" if T4 has room
LANG = "bengali"
N_TRAIN, N_EVAL = 1500, 150            # smaller = safe on Colab RAM; raise later if stable
WB_SUBSETS = ["WestBengal_Kolkata", "WestBengal_North24Parganas", "WestBengal_Malda",
              "WestBengal_PaschimMedinipur", "WestBengal_Purulia", "WestBengal_Jalpaiguri"]

processor = WhisperProcessor.from_pretrained(BASE, language=LANG, task="transcribe")

def lowpass(w, sr, cutoff):
    b, a = butter(6, cutoff / (sr / 2), btype="low")
    return lfilter(b, a, w).astype(np.float32)

# Degrade clean audio to match the client's 4 files (measured: ~99% energy <1kHz, none >3kHz).
def degrade(wav, sr=16000, p=0.8):
    if random.random() > p:
        return wav.astype(np.float32)
    w = lowpass(wav.astype(np.float32), sr, random.uniform(1200, 3400))   # narrow band
    tgt = random.choice([8000, 8000, 4000])                              # telephone / worse
    w = librosa.resample(w, orig_sr=sr, target_sr=tgt)
    w = librosa.resample(w, orig_sr=tgt, target_sr=sr)
    snr = random.uniform(3, 18)                                          # add noise
    rms = np.sqrt(np.mean(w ** 2)) + 1e-9
    return (w + np.random.randn(len(w)).astype(np.float32) * rms / (10 ** (snr / 20))).astype(np.float32)

# %% 4. build data (stream Vaani -> degrade -> features, memory-safe)
from datasets import load_dataset, Audio, Dataset

def build(n):
    feats = []
    for sub in WB_SUBSETS:
        ds = load_dataset("ARTPARK-IISc/Vaani", sub, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        for ex in ds:
            if ex.get("language") != "Bengali" or not ex.get("isTranscriptionAvailable") \
               or not ex.get("transcript"):
                continue
            arr = ex["audio"]["array"]
            if not (1.0 < len(arr) / 16000.0 < 25):
                continue
            wav = degrade(arr)                                  # degrade now, keep only features
            feat = processor.feature_extractor(wav, sampling_rate=16000).input_features[0]
            feats.append({"input_features": feat,
                          "labels": processor.tokenizer(ex["transcript"].strip()).input_ids})
            print(f"  [{sub}] {len(feats)}/{n}", end="\r")
            if len(feats) >= n:
                return feats
    return feats

allf = build(N_TRAIN + N_EVAL)
random.shuffle(allf)
eval_ds = Dataset.from_list(allf[:N_EVAL])
train_ds = Dataset.from_list(allf[N_EVAL:])
del allf; gc.collect()
print(f"\nbuilt {len(train_ds)} train + {len(eval_ds)} eval")

# %% 5. collator + WER metric
import evaluate
from dataclasses import dataclass
wer_metric = evaluate.load("wer")

@dataclass
class Collator:
    processor: object
    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labs = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        batch["labels"] = labs["input_ids"].masked_fill(labs.attention_mask.ne(1), -100)
        return batch

def compute_metrics(pred):
    pred_ids, label_ids = pred.predictions, pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# %% 6. model + LoRA
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained(BASE)
model.generation_config.language = LANG
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none"))
model.print_trainable_parameters()

# %% 7. train (this is the long cell; WER before->after shows in the table)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(
    output_dir="whisper-bn-vaani", per_device_train_batch_size=8, gradient_accumulation_steps=2,
    learning_rate=1e-3, warmup_steps=50, num_train_epochs=3, fp16=True,
    eval_strategy="epoch", save_strategy="epoch", predict_with_generate=True,
    generation_max_length=225, logging_steps=25, report_to="none", remove_unused_columns=False)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=Collator(processor), compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor)
trainer.train()

# %% 8. save
model.save_pretrained("whisper-bn-vaani-lora"); processor.save_pretrained("whisper-bn-vaani-lora")
print("Done. Compare the epoch-1 WER vs the final-epoch WER in the table above (lower = better).")

# %% 9. TEST on your 4 real client files
from google.colab import files
print("Select your 4 audio files:")
up = files.upload()
model.eval()
for fn in up:
    wav, _ = librosa.load(fn, sr=16000)
    feat = processor.feature_extractor(wav, sampling_rate=16000,
                                       return_tensors="pt").input_features.to(model.device)
    with torch.no_grad():
        ids = model.generate(feat, max_new_tokens=225)
    print(f"\n===== {fn} =====")
    print(processor.batch_decode(ids, skip_special_tokens=True)[0])

# %% 10. download the model for on-prem deployment
# !zip -r whisper-bn-vaani-lora.zip whisper-bn-vaani-lora
# from google.colab import files; files.download("whisper-bn-vaani-lora.zip")
