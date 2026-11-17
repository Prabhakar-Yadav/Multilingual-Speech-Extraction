"""Transcribe + translate audio via Groq's hosted Whisper Large V3 (free API,
far stronger at Bengali than the local small/medium/BanglaASR models).

Needs a free key:  setx GROQ_API_KEY "gsk_..."   (or pass as 3rd arg)
NOTE: this uses the cloud. Great for best-quality results now / proving the
concept. The on-prem hospital build still falls back to the local Bengali model.

    python src/transcribe_groq.py <audio> <out.json> [api_key]
"""
import os, sys, json
from pathlib import Path
from groq import Groq

audio = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "outputs/groq_result.json"
key = (sys.argv[3] if len(sys.argv) > 3 else os.environ.get("GROQ_API_KEY", "")).strip()
if not key:
    sys.exit("ERROR: no Groq key. Get a free one at console.groq.com -> set GROQ_API_KEY or pass as 3rd arg.")

client = Groq(api_key=key)
data = Path(audio).read_bytes()
fname = os.path.basename(audio)

# Bengali transcript (source language)
bn = client.audio.transcriptions.create(
    file=(fname, data), model="whisper-large-v3", language="bn",
    response_format="json", temperature=0.0).text

# Direct English translation (Whisper's translate task)
en = client.audio.translations.create(
    file=(fname, data), model="whisper-large-v3",
    response_format="json", temperature=0.0).text

res = {"model": "groq/whisper-large-v3", "audio": audio,
       "bengali": bn, "detected_language": "bn", "english_text": en, "transcript": bn}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("--- BENGALI ---\n", bn)
print("\n--- ENGLISH ---\n", en)
print("\nsaved", out)
