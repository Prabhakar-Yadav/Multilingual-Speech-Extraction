# VaultStack — Medical-Term Extraction from Doctor–Patient Audio

Extracts medical terms (medicines, diagnoses, procedures, tests) from Bengali/Hindi
doctor–patient audio. Runs **fully on-prem, CPU-only, offline, no API keys** — for
8 GB-RAM hospital PCs.

```
audio → ASR (Whisper) → translate→English (NLLB) → medical NER → JSON of terms
```

> ⚠️ **Read [REPORT.md](REPORT.md) first.** The provided sample `1.mp3.mpeg` is too
> low quality (8 kHz / 16 kbps, band-limited below ~1 kHz) to transcribe — every ASR
> model hallucinates on it. This is an **audio-capture** problem. Translation and
> extraction are proven working on usable input.

## Layout
```
src/      transcribe.py  translate.py  extract.py  run_all.py
          download_model.py  analyze_audio.py
models/   downloaded models (faster-whisper-medium, nllb-200-distilled-600M, biomedical-ner-all)
data/     demo inputs (representative Bengali + English consults)
outputs/  transcript / translation / medical_terms / report JSON
```

## Setup
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# log in once (some models gated): python -c "from huggingface_hub import login; login(token='hf_...')"
# download models into models\ :
.\.venv\Scripts\python.exe src\download_model.py Systran/faster-whisper-medium models\faster-whisper-medium
.\.venv\Scripts\python.exe src\download_model.py facebook/nllb-200-distilled-600M models\nllb-200-distilled-600M
.\.venv\Scripts\python.exe src\download_model.py d4data/biomedical-ner-all models\biomedical-ner-all
```

## Run
```powershell
# full pipeline on an audio file
.\.venv\Scripts\python.exe src\run_all.py <audio.wav> --model models\faster-whisper-medium

# check audio quality before trusting a transcript
.\.venv\Scripts\python.exe src\analyze_audio.py <audio.wav>

# stage 2 + 3 on the bundled Bengali demo (no audio needed)
.\.venv\Scripts\python.exe src\translate.py --in data\demo_transcript_bn.json --out outputs\demo_translation.json
.\.venv\Scripts\python.exe src\extract.py   --in outputs\demo_translation.json --out outputs\demo_chain_terms.json
```

## Notes
- `run_all.py` runs each stage in its **own process** so peak RAM ≈ the biggest single
  stage (fits 8 GB). `medium` needs ~4 GB free; a Bengali-fine-tuned `small` is lighter.
- Lightweight NER is a baseline; **MedGemma 4B** is the optional accuracy upgrade
  (accept its license on HuggingFace, then wire `--engine medgemma`).
