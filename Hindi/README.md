# Hindi — Fine-Tuning Practice Run

**Purpose (per the lead):** repeat the whole fine-tuning workflow in **Hindi** to build
confidence in the technique *before* the client's real audio arrives. It also doubles as
real progress — the system is meant to support **Bengali + Hindi**, so this is a usable
Hindi medical-term extractor, not throwaway.

**This folder is self-contained and does NOT touch the main Bengali/English pipeline** —
those models and `src/` scripts are untouched.

## What it does
Hindi doctor-patient transcript (Devanagari) → **English** medical terms JSON
(`medicines, diagnoses, procedures, tests, symptoms`). Same model + technique as Phase 1.

## Files
```
Hindi/
├── gen_data_hindi.py     # makes the synthetic Hindi dataset (DONE)
├── data/train.jsonl      # 1080 Hindi training examples + val.jsonl
├── finetune_colab.py     # Unsloth QLoRA training (run on Colab GPU)
├── models/               # put the downloaded hindi-medical.gguf here
└── README.md
```

## Steps (same loop as the main project)
1. **Data:** already generated. (Regenerate/enlarge with `python Hindi/gen_data_hindi.py 2000`.)
2. **Train on Colab** (free T4 GPU): upload `Hindi/data/train.jsonl`, run `finetune_colab.py`
   (first cell `!pip install unsloth`).
3. **Download** the exported GGUF → `Hindi/models/hindi-medical.gguf`.
4. **Test on-prem** with the existing extractor — point it at the Hindi model:
   ```powershell
   $env:VAULT_GGUF="...\VaultStack\Hindi\models\hindi-medical.gguf"
   python src\extract_local.py "डॉक्टर ने मेटफॉर्मिन दिया, मरीज़ को बुखार है, ब्लड टेस्ट कराना है" out.json
   ```
   (The extractor is language-agnostic — same script, just a different model.)

## Notes
- Base model: `Llama-3.2-3B-Instruct` (decent Hindi). If Hindi quality is weak, switch to
  `Qwen2.5-3B-Instruct` (stronger multilingual) — only the `model_name` line changes.
- Runs on the same on-prem stack: ~4 GB RAM, CPU, offline, via the bundled `llama.cpp` binary.
