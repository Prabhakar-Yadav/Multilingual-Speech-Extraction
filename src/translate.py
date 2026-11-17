"""Phase B - Translate the transcript to English with NLLB-200 (non-gated).

Reads outputs/transcript.json (Phase A). English passes through untouched;
Bengali/Hindi are translated with facebook/nllb-200-distilled-600M on CPU.
(IndicTrans2 is the higher-quality alternative but is gated - swap it in later
if you accept its license and have an HF token.)
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_LOCAL = Path(__file__).resolve().parent.parent / "models" / "nllb-200-distilled-600M"
MODEL_NAME = str(_LOCAL) if _LOCAL.exists() else "facebook/nllb-200-distilled-600M"
SRC_CODE = {"bn": "ben_Beng", "hi": "hin_Deva"}   # whisper code -> NLLB flores code
TGT_CODE = "eng_Latn"


def split_sentences(text: str):
    parts = re.split(r"(?<=[।\.\?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def translate(transcript: str, src_lang: str, batch_size: int = 8) -> str:
    sentences = split_sentences(transcript)
    if not sentences:
        return ""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_CODE[src_lang])
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    bos = tokenizer.convert_tokens_to_ids(TGT_CODE)

    outputs = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            gen = model.generate(**inputs, forced_bos_token_id=bos, max_length=256, num_beams=5)
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return " ".join(outputs).strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/transcript.json")
    ap.add_argument("--out", default="outputs/translation.json")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    lang, transcript = data["detected_language"], data["transcript"]

    t0 = time.time()
    if lang == "en":
        english = transcript
        print("Language is English -> passthrough (no translation).")
    elif lang in SRC_CODE:
        print(f"Translating {lang} -> en with NLLB ...")
        english = translate(transcript, lang)
    else:
        raise SystemExit(f"Unexpected language '{lang}'; expected bn/hi/en.")
    elapsed = round(time.time() - t0, 1)

    result = {
        "source_language": lang,
        "translator": "passthrough" if lang == "en" else MODEL_NAME,
        "processing_sec": elapsed,
        "english_text": english,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processing time : {elapsed}s\nSaved           : {args.out}\n--- ENGLISH ---\n{english}")
