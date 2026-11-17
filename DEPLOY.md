# VaultStack — On-Prem Deployment Guide

Runs **fully offline on the hospital PC** (8 GB RAM, CPU, **no GPU, no internet, no API**).
PHI never leaves the machine. Peak RAM ≈ **4 GB**.

## What the pipeline does
```
audio → faster-whisper (speech → English) → fine-tuned LLM via llama.cpp → medical terms JSON
```
Stages run **one at a time** (ASR loads, runs, frees memory; then the LLM loads) so peak RAM
is the *bigger single stage* (~4 GB), not the sum.

## What to copy onto the hospital PC (the deployment bundle)
```
VaultStack/
├── src/                                  # run_pipeline.py, extract_local.py, transcribe.py
├── requirements_deploy.txt
└── models/
    ├── faster-whisper-medium/            # ASR model  (~1.5 GB)   [or a Bengali-tuned model later]
    ├── llama-3.2-3b-gguf/medical.gguf    # fine-tuned extractor (~1.9 GB)
    └── llamacpp/                         # the llama.cpp Windows CPU binary (llama-server.exe)
```
Total ≈ 3.5 GB on disk. **No torch / transformers needed.**

## One-time install (needs internet ONCE, on a setup machine — then fully offline)
```powershell
# 1. Install Python 3.10 (python.org), then:
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_deploy.txt
```
> For a truly air-gapped install: run `pip download -r requirements_deploy.txt -d wheels\` on an
> online machine, copy the `wheels\` folder, then `pip install --no-index --find-links wheels\ -r requirements_deploy.txt`.

## Run (offline, per audio file)
```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py "path\to\call.wav"  outputs\result.json
```
Output `result.json` contains the English transcript + `medical_terms`
(`medicines, diagnoses, procedures, tests, symptoms`) + timing + RAM used.

## Hardware / performance
| | value |
|---|---|
| RAM peak | ~4 GB (fits 8 GB with headroom) |
| GPU | none required |
| Internet | none required after install |
| Extraction (LLM) | ~15–30 s/file |
| ASR | depends on model + audio length |

## Notes & upgrades
- **CPU compatibility:** the bundled llama.cpp build auto-detects CPU features (works on AVX2
  chips without AVX512). Don't use the `llama-cpp-python` pip wheel — it assumes AVX512 and crashes.
- **Better accuracy** = better recording quality (≥16 kHz). The pipeline is only as good as the audio.
- **Improving the model:** retrain on richer synthetic data (`src/gen_synthetic_data.py`) on a GPU
  (Colab), export GGUF, and drop the new file in `models/llama-3.2-3b-gguf/`. No code changes.
- **Bengali ASR upgrade (Phase 2):** swap `models/faster-whisper-medium` for a Bengali-fine-tuned
  CTranslate2 model when available — `run_pipeline.py` needs no change (set `VAULT_ASR` env var).
