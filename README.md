# Multilingual Speech Extraction

An experimental toolkit for transcribing Bengali, Hindi, and English doctor–patient audio and extracting medical terms such as medicines, diagnoses, procedures, tests, and symptoms.

The repository contains several approaches at different stages of development. It is a research/demo project, not a validated clinical product. Transcription and extraction can be wrong, especially with noisy or low-quality recordings; results need human review.

## Pipeline options

### Local staged pipeline

`src/run_all.py` runs three local stages in sequence:

1. Speech recognition with faster-whisper.
2. Bengali/Hindi to English translation with NLLB (English passes through).
3. Medical entity extraction with the biomedical NER model.

Each stage can also be run separately. See [DEPLOY.md](DEPLOY.md) for the offline deployment variant and its llama.cpp extractor.

### Other experiments

- `src/` contains local and hosted extraction/transcription scripts, audio utilities, and pipeline variants.
- `Bengali/` and `Bengali/ASR/` contain Bengali fine-tuning, evaluation, and ASR experiments.
- `Hindi/` contains Hindi data generation, fine-tuning, and evaluation experiments.
- `English ASR/` contains English benchmark scripts.
- `REPORT.md` summarizes earlier experiments and known limitations. Treat measurements there as project notes, not independent validation.
- `RUN_COMMANDS.md` documents a Bengali training workflow.

## Requirements

- Python 3.10 is the version used in the project instructions.
- A CPU can run the baseline, though larger models may be slow and need several gigabytes of free RAM.
- Downloaded model files are not included in this repository. Keep them under `models/` (or the language-specific model folders) as required by the selected script.
- The full baseline in `requirements.txt` includes PyTorch and Transformers for translation and extraction. `requirements_deploy.txt` is for the lighter llama.cpp deployment path; see [DEPLOY.md](DEPLOY.md).

## Install

From the repository root, on Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux or macOS, use `python3.10 -m venv .venv` and `.venv/bin/python` in place of the Windows commands.

The first run may download model files from their model providers. Review and accept any applicable model licenses before downloading or using a model. For an offline deployment, download all required models and packages before disconnecting; see [DEPLOY.md](DEPLOY.md).

## Run the staged baseline

```powershell
# Transcribe audio, then translate and extract terms.
.\.venv\Scripts\python.exe src\run_all.py "path\to\audio.wav" --model small

# Output files are written under outputs/.
```

You can also run individual steps:

```powershell
.\.venv\Scripts\python.exe src\transcribe.py "path\to\audio.wav" --model small --out outputs\transcript.json
.\.venv\Scripts\python.exe src\translate.py --in outputs\transcript.json --out outputs\translation.json
.\.venv\Scripts\python.exe src\extract.py --in outputs\translation.json --out outputs\medical_terms.json
```

Use `src/analyze_audio.py` to inspect recording quality before interpreting a transcript. Poor source audio cannot be repaired reliably by a larger model.

## Data, privacy, and generated files

Do not commit real patient recordings, transcripts, identifiers, credentials, or other confidential material. Keep private inputs in a local directory such as `data/private/`; it is ignored by Git. Model weights, local environment files, logs, training checkpoints, and generated outputs are also ignored.

Some example and experiment data files are already tracked in this repository. Review their contents and confirm you have permission to publish them before redistributing this project. The ignore rules do not remove files that Git already tracks.

## Project notes

- The pipeline is experimental and may produce missed, incorrect, or invented terms. Have a qualified person review every result before any use in care or operations.
- The bundled scripts use different models and runtime assumptions; check each script or its folder README before running an experiment.
- Audio and transcript files can contain sensitive health information. Process them only in an environment approved for that data, and avoid storing generated transcripts or logs unnecessarily.

## License

No project license is currently specified. Until one is added, reuse and redistribution of this repository's code and data are not granted by an open-source license. Model licenses are separate and must also be followed.
