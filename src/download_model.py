"""Robust model downloader for flaky/slow connections.

huggingface_hub's snapshot_download stalls on this network (Xet client hangs;
even the HTTP fallback wouldn't pull the big file). A plain streamed GET works
(~250 KB/s), so we do that ourselves with Range-based resume + infinite retry,
saving into a local folder that faster-whisper / transformers can load by path.

    python src/download_model.py <repo_id> <dest_dir>
"""
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi, hf_hub_url, get_token

repo = sys.argv[1] if len(sys.argv) > 1 else "Systran/faster-whisper-medium"
dest_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "models/faster-whisper-medium")
dest_dir.mkdir(parents=True, exist_ok=True)

token = get_token()
auth = {"Authorization": f"Bearer {token}"} if token else {}

skip_ext = (".gitattributes", ".mp3", ".wav", ".flac")
skip_names = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin",
              "trainer_state.json", "README.md"}
files = [f for f in HfApi().list_repo_files(repo)
         if not f.endswith(skip_ext) and Path(f).name not in skip_names]
# If safetensors weights exist, skip the duplicate pytorch_model.bin.
if any(f.endswith("model.safetensors") for f in files):
    files = [f for f in files if not f.endswith("pytorch_model.bin")]
# optional 3rd arg: only fetch files whose name contains this substring (e.g. a GGUF quant)
name_filter = sys.argv[3] if len(sys.argv) > 3 else None
if name_filter:
    files = [f for f in files if name_filter.lower() in f.lower()]
print("Files to fetch:", files, flush=True)


def fetch(fname: str):
    url = hf_hub_url(repo, fname)
    dest = dest_dir / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        h = dict(auth)
        if have:
            h["Range"] = f"bytes={have}-"
        try:
            with httpx.stream("GET", url, headers=h, follow_redirects=True, timeout=60) as r:
                if r.status_code == 416:           # range past EOF => already complete
                    print(f"  {fname}: already complete ({have/1e6:.1f} MB)", flush=True)
                    return
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                done, last = have, time.time()
                with open(dest, mode) as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 15:
                            pct = f"{done/total*100:.0f}%" if total else "?"
                            print(f"  {fname}: {done/1e6:.1f}/{total/1e6:.1f} MB ({pct})", flush=True)
                            last = time.time()
            final = dest.stat().st_size
            if total and final < total:
                print(f"  {fname}: dropped at {final/1e6:.1f} MB, resuming...", flush=True)
                continue
            print(f"  {fname}: DONE {final/1e6:.1f} MB", flush=True)
            return
        except Exception as e:
            print(f"  {fname}: {type(e).__name__}: {e} -- retry in 3s", flush=True)
            time.sleep(3)


for fn in files:
    fetch(fn)
print("ALL_DONE", dest_dir, flush=True)
