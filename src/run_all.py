"""Phase D - Run the whole pipeline on one audio file.

Each stage runs as a SEPARATE process so its model is fully released from RAM
before the next stage loads - that keeps peak memory ~= the largest single
stage (not the sum), which is what lets this fit the 8 GB target machine.

    python src/run_all.py <audio> [--model small|medium]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path("outputs")


def run(stage, *cmd):
    print(f"\n===== {stage} =====")
    t0 = time.time()
    r = subprocess.run([sys.executable, *cmd])
    if r.returncode != 0:
        raise SystemExit(f"{stage} failed (exit {r.returncode}).")
    return round(time.time() - t0, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="small")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    timings = {}
    timings["asr"] = run("Stage 1: ASR", str(HERE / "transcribe.py"), args.audio,
                         "--model", args.model, "--out", str(OUT / "transcript.json"))
    timings["translate"] = run("Stage 2: Translate", str(HERE / "translate.py"),
                                "--in", str(OUT / "transcript.json"),
                                "--out", str(OUT / "translation.json"))
    timings["extract"] = run("Stage 3: Extract", str(HERE / "extract.py"),
                             "--in", str(OUT / "translation.json"),
                             "--out", str(OUT / "medical_terms.json"))

    # Merge into one report.
    tr = json.loads((OUT / "transcript.json").read_text(encoding="utf-8"))
    tl = json.loads((OUT / "translation.json").read_text(encoding="utf-8"))
    ex = json.loads((OUT / "medical_terms.json").read_text(encoding="utf-8"))
    report = {
        "audio": args.audio,
        "language": tr["detected_language"],
        "asr_model": tr["model"],
        "transcript": tr["transcript"],
        "english_text": tl["english_text"],
        "medical_terms": ex["terms"],
        "timings_sec": timings,
        "total_sec": round(sum(timings.values()), 1),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== DONE =====")
    print(f"Timings (s): {timings}  total={report['total_sec']}s")
    print(f"Report: {OUT / 'report.json'}")
    print("\nMedical terms to report:")
    print(json.dumps(report["medical_terms"], ensure_ascii=False, indent=2))
