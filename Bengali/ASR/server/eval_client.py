"""
Test the fine-tuned Bengali ASR on the CLIENT's 4 real calls.

TEST ONLY. Nothing here is ever fed back into training - the client transcripts
are our single independent yardstick, so contaminating them would destroy the
only honest signal we have.

    python eval_client.py                       # newest checkpoint
    CKPT=ckpt_run2/checkpoint-2846 python eval_client.py
    BASE_ONLY=1 python eval_client.py           # un-finetuned baseline, for comparison

Each audio file is scored against the gold call it matches best (min WER), the
same rule the earlier eval used - the 4 files and 3 gold calls aren't 1:1.
"""
import json, os, time
from pathlib import Path

import jiwer, torch
from peft import PeftModel
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          pipeline)

HOME  = Path.home() / "bengali_asr"
AUDIO = HOME / "client_audio"
GOLD  = json.loads((AUDIO / "gold.json").read_text(encoding="utf-8"))
MODEL = "openai/whisper-large-v3"
BASE_ONLY = os.environ.get("BASE_ONLY") == "1"


def newest_ckpt():
    cks = sorted((HOME / "ckpt_run2").glob("checkpoint-*"),
                 key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


CKPT = os.environ.get("CKPT")
CKPT = Path(CKPT) if CKPT else newest_ckpt()

print(f"model : {MODEL}")
print(f"adapter: {'NONE (baseline)' if BASE_ONLY else CKPT}", flush=True)

processor = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16).to("cuda")
if not BASE_ONLY:
    model = PeftModel.from_pretrained(model, str(CKPT)).merge_and_unload()
model.eval()

# chunk_length_s: the calls are minutes long, far past Whisper's 30s window
asr = pipeline("automatic-speech-recognition", model=model,
               tokenizer=processor.tokenizer,
               feature_extractor=processor.feature_extractor,
               chunk_length_s=30, stride_length_s=5,
               torch_dtype=torch.float16, device="cuda")


def collapse(t):
    """Whisper sometimes loops a word; collapse runs so one glitch isn't N errors."""
    out = []
    for w in t.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)


rows, wers, cers = [], [], []
for f in sorted(AUDIO.glob("*.mpeg")) + sorted(AUDIO.glob("*.mp3")):
    t0 = time.time()
    hyp = collapse(asr(str(f), generate_kwargs={"language": "bengali",
                                                "task": "transcribe"})["text"].strip())
    took = time.time() - t0

    best = min(GOLD, key=lambda g: jiwer.wer(g["bengali"], hyp))
    w = 100 * jiwer.wer(best["bengali"], hyp)
    c = 100 * jiwer.cer(best["bengali"], hyp)
    wers.append(w); cers.append(c)
    rows.append({"file": f.name, "call": best["call"], "patient": best["patient"],
                 "wer": round(w, 1), "cer": round(c, 1), "hyp": hyp})
    print(f"  {f.name:16} -> Call {best['call']} ({best['patient']:15}) "
          f"WER {w:5.1f}%  CER {c:5.1f}%  [{took:.0f}s]", flush=True)

print("\n" + "=" * 64)
print(f"  CLIENT CALLS  |  adapter: {'baseline' if BASE_ONLY else CKPT.name}")
print("=" * 64)
print(f"  avg WER {sum(wers)/len(wers):5.1f}%   (word accuracy {100-sum(wers)/len(wers):5.1f}%)")
print(f"  avg CER {sum(cers)/len(cers):5.1f}%   (char accuracy {100-sum(cers)/len(cers):5.1f}%)")
print("=" * 64)
print("  WER/CER are ERROR rates - LOWER IS BETTER.")

out = HOME / f"client_eval_{'baseline' if BASE_ONLY else CKPT.name}.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n  transcripts saved -> {out}")
