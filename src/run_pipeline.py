"""FULL on-prem pipeline (offline, CPU): audio -> faster-whisper (translate to
English) -> local fine-tuned LLM (llama.cpp) -> medical terms JSON.
Stages run sequentially so peak RAM = the bigger single stage (fits 8 GB).

    python src/run_pipeline.py <audio_file> [out.json]
"""
import os, sys, json, time, tempfile, subprocess
from pathlib import Path
from faster_whisper import WhisperModel

B = Path(__file__).resolve().parent.parent
ASR_MODEL = os.environ.get("VAULT_ASR") or str(B / "models" / "faster-whisper-medium")
GGUF = os.environ.get("VAULT_GGUF") or str(B / "models" / "llama-3.2-3b-gguf" / "medical.gguf")
PY = sys.executable


def main():
    audio = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/result.json"
    t0 = time.time()

    # Stage 1 — ASR + translate to English (on-prem, CPU, no GPU)
    model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    segs, info = model.transcribe(audio, task="translate", vad_filter=True,
                                  beam_size=1, condition_on_previous_text=False)
    english = " ".join(s.text.strip() for s in segs).strip()
    dur, asr_t = round(info.duration, 1), round(time.time() - t0, 1)
    del model                                          # release ASR RAM before the LLM loads

    # Stage 2 — extract terms with the local fine-tuned LLM (own process => clean RAM)
    tmp = Path(tempfile.gettempdir()) / "_vault_en.json"
    tmp.write_text(json.dumps({"translation_english": english}), encoding="utf-8")
    termsfile = Path(tempfile.gettempdir()) / "_vault_terms.json"
    env = dict(os.environ); env["VAULT_GGUF"] = GGUF
    subprocess.run([PY, str(B / "src" / "extract_local.py"), str(tmp), str(termsfile)],
                   env=env, check=True)
    ex = json.loads(termsfile.read_text(encoding="utf-8"))

    result = {
        "file": Path(audio).name,
        "audio_duration_sec": dur,
        "asr_time_sec": asr_t,
        "extract_time_sec": ex.get("processing_sec"),
        "extract_ram_gb": ex.get("llm_process_rss_gb"),
        "translation_english": english,
        "medical_terms": ex.get("terms", {}),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["medical_terms"], ensure_ascii=False, indent=2))
    print(f"\nASR {asr_t}s | total {round(time.time()-t0,1)}s | RAM peak ~{ex.get('llm_process_rss_gb')}GB | saved {out}")


if __name__ == "__main__":
    main()
