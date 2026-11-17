"""Make-or-break test: can we download ONE Vaani West-Bengal shard locally and
does it contain transcribed Bengali? If this works, local training is viable.
"""
import os, collections
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

SHARD = "audio/WestBengal/Kolkata/train-00000-of-00047.parquet"
print("downloading (resumable):", SHARD, flush=True)
p = hf_hub_download("ARTPARK-IISc/Vaani", SHARD, repo_type="dataset")
print("OK ->", round(os.path.getsize(p) / 1e6, 1), "MB @", p, flush=True)

sch = pq.read_schema(p)
print("columns:", sch.names, flush=True)
meta_cols = [c for c in ["language", "isTranscriptionAvailable", "transcript"] if c in sch.names]
t = pq.read_table(p, columns=meta_cols)                       # metadata only, skips audio -> fast
langs = collections.Counter(t.column("language").to_pylist()) if "language" in sch.names else {}
print("rows in shard:", t.num_rows, flush=True)
print("languages:", langs.most_common(8), flush=True)
if "language" in sch.names and "isTranscriptionAvailable" in sch.names:
    bn = sum(1 for l, tr in zip(t.column("language").to_pylist(),
                                t.column("isTranscriptionAvailable").to_pylist())
             if l == "Bengali" and tr)
    print("Bengali WITH transcript in this one shard:", bn, flush=True)
