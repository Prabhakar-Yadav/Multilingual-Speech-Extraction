"""Download the official llama.cpp prebuilt Windows CPU binary (runtime CPU-feature
dispatch -> safe on AVX2-only chips, unlike the AVX512 pip wheel). Unzips to
models/llamacpp/."""
import io, zipfile
from pathlib import Path
import httpx

B = Path(__file__).resolve().parent.parent
dest = B / "models" / "llamacpp"
dest.mkdir(parents=True, exist_ok=True)

rel = httpx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                timeout=60, follow_redirects=True).json()
assets = rel.get("assets", [])
names = [a["name"] for a in assets]
print("release:", rel.get("tag_name"))

def pick():
    for a in assets:
        n = a["name"].lower()
        if "win" in n and "x64" in n and "cpu" in n and n.endswith(".zip"):
            return a
    for a in assets:                       # fallback: any win x64 zip
        n = a["name"].lower()
        if "win" in n and "x64" in n and n.endswith(".zip"):
            return a
    return None

a = pick()
if not a:
    raise SystemExit("no win-x64 cpu asset found; assets=" + str(names))
print("downloading:", a["name"])
data = io.BytesIO()
with httpx.stream("GET", a["browser_download_url"], follow_redirects=True, timeout=120) as r:
    r.raise_for_status()
    for chunk in r.iter_bytes(1 << 20):
        data.write(chunk)
zipfile.ZipFile(data).extractall(dest)
exes = [str(p.relative_to(dest)) for p in dest.rglob("*.exe")]
print("extracted exes:", exes)
print("DONE", dest)
