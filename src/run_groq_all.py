"""Hybrid robust runner: all 4 audio files through Groq Whisper Large V3
(transcribe bn + translate en) then Groq Llama-3.3-70B medical extraction.
Saves per-file groq_result.json + groq_terms.json and prints a summary.

    python src/run_groq_all.py [api_key]      (or set GROQ_API_KEY)
"""
import os, sys, json, glob
from pathlib import Path
from groq import Groq

B = Path(__file__).resolve().parent.parent
key = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GROQ_API_KEY", "")).strip()
if not key:
    sys.exit("ERROR: pass Groq key as arg or set GROQ_API_KEY (free at console.groq.com)")

client = Groq(api_key=key)
SYS = ("You are a clinical NLP extractor. Input is a noisy Bengali->English "
       "doctor-patient phone transcript. Extract ONLY medical terms genuinely "
       "present as STRICT JSON: medicines, diagnoses, procedures, tests, symptoms "
       "(string arrays). Normalize real drug names; empty arrays if garbled/absent. "
       "NEVER invent drug names.")

EXTS = (".mp3", ".wav", ".mpeg", ".m4a", ".ogg", ".flac")
files = sorted(f for f in glob.glob(str(B / "data" / "*")) if f.lower().endswith(EXTS))
print("FILES:", [Path(f).name for f in files], flush=True)

for f in files:
    n = Path(f).name
    folder = B / "outputs" / n.split(".")[0]
    folder.mkdir(parents=True, exist_ok=True)
    try:
        data = Path(f).read_bytes()
        bn = client.audio.transcriptions.create(file=(n, data), model="whisper-large-v3",
                                                 language="bn", response_format="json", temperature=0).text
        en = client.audio.translations.create(file=(n, data), model="whisper-large-v3",
                                               response_format="json", temperature=0).text
        (folder / "groq_result.json").write_text(json.dumps(
            {"model": "groq/whisper-large-v3", "bengali": bn, "english_text": en},
            ensure_ascii=False, indent=2), encoding="utf-8")
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": en}],
            response_format={"type": "json_object"}, temperature=0)
        terms = json.loads(resp.choices[0].message.content)
        (folder / "groq_terms.json").write_text(json.dumps(
            {"engine": "groq/llama-3.3-70b", "terms": terms}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== {n} ===\nEN: {en[:240]}\nTERMS: {json.dumps(terms, ensure_ascii=False)}", flush=True)
    except Exception as ex:
        print(f"\n=== {n} === ERROR: {ex}", flush=True)
print("\nALL_DONE", flush=True)
