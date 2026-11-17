"""Robust batch runner: each file, each stage in its OWN subprocess so memory is
fully released between steps (the one-process version OOM'd). Continues on error.
"""
import subprocess, sys, glob, json
from pathlib import Path

B = Path(__file__).resolve().parent.parent
PY = str(B / ".venv" / "Scripts" / "python.exe")
S, O = B / "src", B / "outputs"
EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac", ".mpeg", ".aac")
files = sorted(f for f in glob.glob(str(B / "data" / "*")) if f.lower().endswith(EXTS))
print("FILES:", [Path(f).name for f in files], flush=True)


def run(args):
    return subprocess.run([PY] + [str(a) for a in args], capture_output=True, text=True)


for f in files:
    n = Path(f).name
    try:
        cw = O / f"clean_{n}.wav"
        run([S / "clean_audio.py", f, cw])
        tj = O / f"bn_{n}.json"
        run([S / "transcribe_hf.py", B / "models" / "banglaasr", cw, tj])
        cj = O / f"bnclean_{n}.json"
        run([S / "clean_transcript.py", tj, cj])
        trj = O / f"tr_{n}.json"
        run([S / "translate.py", "--in", cj, "--out", trj])
        mj = O / f"result_{n}.json"
        run([S / "extract.py", "--in", trj, "--out", mj])
        terms = json.loads(Path(mj).read_text(encoding="utf-8")).get("terms", {})
        en = json.loads(Path(trj).read_text(encoding="utf-8")).get("english_text", "")[:200]
        print(f"\n=== {n} ===\nEN: {en}\nTERMS: {json.dumps(terms, ensure_ascii=False)}", flush=True)
    except Exception as ex:
        print(f"\n=== {n} === ERROR: {ex}", flush=True)
print("\nALL_DONE", flush=True)
