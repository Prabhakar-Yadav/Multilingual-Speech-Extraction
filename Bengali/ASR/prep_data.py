"""STEP 1 (no torch): read every Vaani Kolkata shard already on disk with pyarrow,
decode + resample audio, CLEAN the transcripts (strip <pause>/<noise>/[..]/{..} tags),
save clean 16kHz Bengali waveforms + transcripts to a pickle.

Kept torch-free because torch + pyarrow-parquet-read segfaults together on this machine.
Auto-picks up new shards as get_shards.py downloads them.

    .\.venv-train\Scripts\python.exe -u Bengali_ASR\prep_data.py
"""
import io, re, pickle, glob, numpy as np, soundfile as sf
from pathlib import Path
from math import gcd
from scipy.signal import resample_poly
import pyarrow.parquet as pq

B = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache/huggingface/hub/datasets--ARTPARK-IISc--Vaani/snapshots"
# RURAL SOUTH BENGAL dialect (matches the client's calls) - NOT urban Kolkata
DISTRICTS = ["North24Parganas", "PaschimMedinipur", "Jhargram"]
shards = []
for d in DISTRICTS:
    shards += sorted(glob.glob(str(CACHE / f"*/audio/WestBengal/{d}/train-*.parquet")))
print(f"found {len(shards)} DIALECT shard(s): {DISTRICTS}", flush=True)

TAG = re.compile(r"<[^>]*>|\[[^\]]*\]|\{[^}]*\}")

def clean(t):
    t = TAG.sub(" ", t).replace("--", " ")
    return re.sub(r"\s+", " ", t).strip()

rows = []
for sh in shards:
    pf = pq.ParquetFile(sh)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["audio", "language", "transcript", "isTranscriptionAvailable"])
        auds = t.column("audio").to_pylist(); langs = t.column("language").to_pylist()
        trs = t.column("transcript").to_pylist(); avail = t.column("isTranscriptionAvailable").to_pylist()
        for a, l, tr, av in zip(auds, langs, trs, avail):
            if l != "Bengali" or not av or not tr:
                continue
            text = clean(str(tr))
            if len(text) < 4 or len(text.split()) < 2:      # skip near-empty after cleaning
                continue
            try:
                wav, sr = sf.read(io.BytesIO(a["bytes"]))
            except Exception:
                continue
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                g = gcd(int(sr), 16000); wav = resample_poly(wav, 16000 // g, int(sr) // g).astype(np.float32)
            if not (1.0 < len(wav) / 16000 < 25):
                continue
            rows.append({"wav": wav, "text": text})
    print(f"  {Path(sh).name}: total clips now {len(rows)}", flush=True)

out = B / "data" / "bn_clips_dialect.pkl"      # rural dialect set (keeps the Kolkata one intact)
out.parent.mkdir(exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(rows, f)
print("SAVED", len(rows), "clean clips ->", out, flush=True)
