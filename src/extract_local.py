"""On-prem medical-term extraction via a LOCAL llama.cpp server (CPU, offline).
Launches llama-server with the quantized GGUF, calls its OpenAI-compatible API,
then shuts it down. Replaces the Groq cloud call. Reports terms + peak RAM.

    python src/extract_local.py <input.json|text> [out.json]
"""
import os, sys, json, time, socket, subprocess
from pathlib import Path
import httpx
import psutil

B = Path(__file__).resolve().parent.parent
# override with env var VAULT_GGUF to use the fine-tuned model
GGUF = os.environ.get("VAULT_GGUF") or str(B / "models" / "llama-3.2-3b-gguf" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

SYS = ("You are a clinical NLP extractor for doctor-patient call transcripts. "
       "Extract ONLY medical terms actually present in the text. Return STRICT JSON "
       "with exactly these keys: medicines, diagnoses, procedures, tests, symptoms - "
       "each an array of strings. Empty arrays if none. Do NOT invent terms.")


def find_server():
    for p in (B / "models" / "llamacpp").rglob("llama-server.exe"):
        return str(p)
    raise SystemExit("llama-server.exe not found - run src/get_llamacpp.py")


def load_text(arg):
    p = Path(arg)
    if p.exists() and p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("translation_english") or d.get("english_text") or ""
    return p.read_text(encoding="utf-8") if p.exists() else arg


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def main():
    text = load_text(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else None
    port = free_port()
    srv = subprocess.Popen([find_server(), "-m", GGUF, "--host", "127.0.0.1",
                            "--port", str(port), "-c", "4096", "-t", "4"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    peak = 0.0
    try:
        for _ in range(180):                       # wait for model load
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        t0 = time.time()
        r = httpx.post(base + "/v1/chat/completions", timeout=600, json={
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 512,
            "response_format": {"type": "json_object"}})
        elapsed = round(time.time() - t0, 1)
        try:
            peak = round(psutil.Process(srv.pid).memory_info().rss / 1e9, 2)
        except Exception:
            pass
        content = r.json()["choices"][0]["message"]["content"]
        # models sometimes wrap JSON in ``` fences or prose; grab the {...} span
        if "{" in content and "}" in content:
            content = content[content.find("{"): content.rfind("}") + 1]
        try:
            terms = json.loads(content)
        except Exception:
            terms = {"_raw": content}
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()

    result = {"engine": "local/llama-3.2-3b-instruct-q4_k_m (llama.cpp server, CPU)",
              "processing_sec": elapsed, "llm_process_rss_gb": peak, "terms": terms}
    if out:
        Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(terms, ensure_ascii=False, indent=2))
    print(f"\nLLM RAM (server RSS): {peak} GB   |   extraction time: {elapsed}s")


if __name__ == "__main__":
    main()
