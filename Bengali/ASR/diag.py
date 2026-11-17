import io, random, numpy as np, torch, librosa, soundfile as sf
from pathlib import Path
from scipy.signal import butter, lfilter
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
print("A imports ok", flush=True)

LOCAL = r"C:\Users\PRABHAKAR\Documents\VaultStack\Bengali_ASR\models\whisper-medium"
print("A1 loading processor...", flush=True)
proc = WhisperProcessor.from_pretrained(LOCAL, language="bengali", task="transcribe")
print("A2 processor ok, getting shard...", flush=True)
p = hf_hub_download("ARTPARK-IISc/Vaani", "audio/WestBengal/Kolkata/train-00000-of-00047.parquet", repo_type="dataset")
print("A3 shard ok", flush=True)

def lowpass(w, sr, c):
    b, a = butter(6, c / (sr / 2), btype="low"); return lfilter(b, a, w).astype(np.float32)
def degrade(wav, sr=16000, prob=0.8):
    if random.random() > prob: return wav.astype(np.float32)
    w = lowpass(wav.astype(np.float32), sr, random.uniform(1200, 3400))
    tgt = random.choice([8000, 8000, 4000]); w = librosa.resample(w, orig_sr=sr, target_sr=tgt)
    w = librosa.resample(w, orig_sr=tgt, target_sr=sr)
    snr = random.uniform(3, 18); rms = np.sqrt(np.mean(w ** 2)) + 1e-9
    return (w + np.random.randn(len(w)).astype(np.float32) * rms / (10 ** (snr / 20))).astype(np.float32)

print("A4 opening parquet...", flush=True)
pf = pq.ParquetFile(p)
t = pf.read_row_group(0, columns=["audio", "language", "transcript", "isTranscriptionAvailable"])
print("A5 row group read", flush=True)
auds = t.column("audio").to_pylist(); langs = t.column("language").to_pylist()
trs = t.column("transcript").to_pylist(); avail = t.column("isTranscriptionAvailable").to_pylist()
print("A6 to_pylist ok, starting loop", flush=True)
feats = []
for a, l, tr, av in zip(auds, langs, trs, avail):
    if l != "Bengali" or not av or not tr: continue
    wav, sr = sf.read(io.BytesIO(a["bytes"])); wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000: wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    if not (1.0 < len(wav) / 16000 < 25): continue
    print(f"  clip {len(feats)}: decoded {len(wav)} samples, degrading...", flush=True)
    dwav = degrade(wav)
    print(f"  clip {len(feats)}: degraded, extracting features...", flush=True)
    feats.append({"input_features": proc.feature_extractor(dwav, sampling_rate=16000).input_features[0],
                  "labels": proc.tokenizer(tr.strip()).input_ids})
    print(f"  clip {len(feats)-1}: done", flush=True)
    if len(feats) >= 5: break
print("B featurized", len(feats), flush=True)

model = WhisperForConditionalGeneration.from_pretrained(
    LOCAL, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0})
print("C model loaded on GPU", flush=True)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none"))
print("D peft wrapped", flush=True)

@dataclass
class Collator:
    processor: object
    def __call__(self, features):
        batch = self.processor.feature_extractor.pad([{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labs = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        batch["labels"] = labs["input_ids"].masked_fill(labs.attention_mask.ne(1), -100)
        return batch

args = Seq2SeqTrainingArguments(output_dir=r"C:\Users\PRABHAKAR\Documents\VaultStack\Bengali_ASR\outputs",
    per_device_train_batch_size=1, gradient_accumulation_steps=4, num_train_epochs=1, fp16=True,
    optim="paged_adamw_8bit", gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1, report_to="none", remove_unused_columns=False, max_steps=2)
tr = Seq2SeqTrainer(model=model, args=args, train_dataset=feats, data_collator=Collator(proc),
    processing_class=proc.feature_extractor)
print("E trainer built, starting 2 steps...", flush=True)
tr.train()
print("F TRAINED 2 STEPS OK", flush=True)
