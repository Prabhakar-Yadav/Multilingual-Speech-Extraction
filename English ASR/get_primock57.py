"""
STEP 1 - Get the benchmark data: PriMock57.

57 mock primary-care consultations (Babylon Health): audio + HUMAN transcripts +
consultation notes. Public, purpose-built as a conversational medical ASR benchmark.
This gives us ground truth, so every WER number we report is honest.

    python get_primock57.py

Needs git (the repo uses git-lfs for the audio).
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = DATA / "primock57"
URL = "https://github.com/babylonhealth/primock57.git"


def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def main():
    DATA.mkdir(exist_ok=True)

    if REPO.exists():
        print(f"already cloned -> {REPO}")
    else:
        print("cloning PriMock57 (audio is git-lfs, this may take a few minutes)...")
        if run(["git", "clone", URL, str(REPO)]).returncode != 0:
            print("\nclone failed. Is git installed? -> https://git-scm.com/downloads")
            sys.exit(1)
        # audio files come down as LFS pointers unless we pull them
        run(["git", "lfs", "pull"], cwd=REPO)

    wavs = list(REPO.rglob("*.wav"))
    txts = list(REPO.rglob("*.txt")) + list(REPO.rglob("*.json"))
    print("\n--- what we got ---")
    print(f"  audio files      : {len(wavs)}")
    print(f"  transcript files : {len(txts)}")

    if wavs:
        import os
        mb = sum(os.path.getsize(w) for w in wavs) / 1e6
        print(f"  total audio      : {mb:.0f} MB")
        print(f"  example          : {wavs[0].relative_to(REPO)}")
    else:
        print("\n  NO AUDIO FOUND - the files are probably still LFS pointers.")
        print("  Install git-lfs (https://git-lfs.com) then run:")
        print(f"    cd {REPO} && git lfs pull")

    print("\nnext:  python benchmark_asr.py")


if __name__ == "__main__":
    main()
