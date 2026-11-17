"""Bulletproof resumable download of openai/whisper-medium (public model) into a
local folder, for a flaky connection. Byte-range resume + endless retries; each
drop continues from the current byte offset instead of restarting. Seeds the big
safetensors from the 256MB already sitting in the HF cache so it's not wasted.

    .\.venv-train\Scripts\python.exe -u Bengali_ASR\get_whisper_medium.py

Then point training at it:  BASE = str(Path(__file__).parent / "models" / "whisper-medium")
"""
import time, shutil, glob
from pathlib import Path
import httpx

OUT = Path(__file__).resolve().parent / "models" / "whisper-medium"
OUT.mkdir(parents=True, exist_ok=True)
BASEURL = "https://huggingface.co/openai/whisper-medium/resolve/main/"
FILES = ["config.json", "generation_config.json", "preprocessor_config.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "normalizer.json", "added_tokens.json", "special_tokens_map.json",
         "model.safetensors"]

# seed the big file from whatever the HF cache already pulled (256MB), if present
seed_part = OUT / "model.safetensors.part"
if not seed_part.exists() and not (OUT / "model.safetensors").exists():
    cands = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--openai--whisper-medium/blobs/*.incomplete"))
    cands = [c for c in cands if Path(c).stat().st_size > 10 * 1024 * 1024]   # >10MB = the real partial
    if cands:
        biggest = max(cands, key=lambda c: Path(c).stat().st_size)
        shutil.copy(biggest, seed_part)
        print(f"seeded model.safetensors.part from cache: {round(seed_part.stat().st_size/1e6,1)} MB", flush=True)


def download(fn):
    dst, part = OUT / fn, OUT / (fn + ".part")
    if dst.exists():
        print(fn, "already done"); return True
    url = BASEURL + fn
    while True:
        pos = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
                if r.status_code == 206:          # resume accepted
                    mode = "ab"
                elif r.status_code == 200:        # server ignored range -> restart file
                    mode, pos = "wb", 0
                else:
                    print(fn, "HTTP", r.status_code); return False
                total = pos + int(r.headers.get("Content-Length", 0))
                last = time.time()
                with open(part, mode) as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        if time.time() - last > 10:
                            print(f"  {fn}: {round(part.stat().st_size/1e6,1)}/{round(total/1e6,1)} MB", flush=True)
                            last = time.time()
            part.rename(dst)
            print(fn, "OK ->", round(dst.stat().st_size / 1e6, 1), "MB", flush=True)
            return True
        except Exception as e:
            got = part.stat().st_size if part.exists() else 0
            print(f"  {fn}: drop at {round(got/1e6,1)} MB, resuming... ({str(e)[:50]})", flush=True)
            time.sleep(3)


for fn in FILES:
    download(fn)
print("ALL DONE ->", OUT, flush=True)
