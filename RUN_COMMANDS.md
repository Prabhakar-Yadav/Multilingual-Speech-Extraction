# Bengali ASR — MAX-accuracy clean-Vaani fine-tune

Goal: lowest WER / highest accuracy on clean Vaani (same-source test).
Training data: ALL 3,298 Vaani clips (dialect + Kolkata), trained on CLEAN audio.
Model: whisper-medium + LoRA (rank 32, 4 attention layers), 3 epochs.

---

## 1. START the training (this is REAL training — takes ~10-12 HOURS)
Also the RESUME command — if it crashes/reboots, run the SAME thing again.

```powershell
cd C:\Users\PRABHAKAR\Documents\VaultStack\Voice-AI
$env:HF_HUB_OFFLINE="1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
..\.venv-train\Scripts\python.exe -u Bengali\ASR\finetune_whisper_local.py 2>&1 | Tee-Object -FilePath train_max_log.txt
```

- Checkpoints every 100 steps -> a crash loses at most ~40 min; re-run to auto-resume.
- VRAM hard-capped (~3.8 GB) -> CANNOT blue-screen the PC.
- Saves final model to `Bengali\ASR\models\whisper-bn-vaani-max`.

## 2. CHECK PROGRESS (second PowerShell window)
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
Get-Content C:\Users\PRABHAKAR\Documents\VaultStack\Voice-AI\train_max_log.txt -Tail 5
```
Watch the `eval_wer` line each epoch — it should drop toward ~20-25%.

## 3. AFTER IT FINISHES — get WORD + CHARACTER accuracy on clean Vaani
First point the eval at the new model, then run it:
```powershell
cd C:\Users\PRABHAKAR\Documents\VaultStack\Voice-AI
(Get-Content Bengali\ASR\eval_cer.py) -replace 'whisper-bn-dialect-lora','whisper-bn-vaani-max' | Set-Content Bengali\ASR\eval_cer.py
$env:HF_HUB_OFFLINE="1"
..\.venv-train\Scripts\python.exe -u Bengali\ASR\eval_cer.py
```

Prints:
```
  Word Error Rate (WER):      XX.X%  ->  word accuracy      XX.X%
  Character Error Rate (CER): XX.X%  ->  CHARACTER accuracy XX.X%
```

---
## Honest expectation
- WORD accuracy: likely ~78-85% (Bengali's hard ceiling is ~76% in papers; more same-source
  data may push a bit past it, but 90% WORD accuracy is very unlikely for Bengali).
- CHARACTER accuracy: should clear 90% — the fair metric for Bengali.
Both numbers are on held-out clean Vaani the model never trained on (honest).
