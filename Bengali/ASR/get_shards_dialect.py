"""Download Vaani shards from the client's ACTUAL dialect region.

Client's doc says: rural South Bengal / Sundarban-adjacent (24 Parganas, Howrah, Hooghly).
Vaani HAS North24Parganas (Sundarban-adjacent) + other rural South Bengal districts.
NOTE: every district has a DIFFERENT shard count - using the wrong one 404s everything
(that's why the first attempt found nothing).
"""
import time
from huggingface_hub import hf_hub_download

# (district, total_shards, how_many_to_pull)
TARGETS = [("North24Parganas", 38, 5),      # exact dialect match (Sundarban-adjacent, rural)
           ("PaschimMedinipur", 68, 2),     # rural South Bengal
           ("Jhargram", 61, 2)]             # rural South Bengal

for d, total, n in TARGETS:
    for i in range(n):
        sh = f"audio/WestBengal/{d}/train-{i:05d}-of-{total:05d}.parquet"
        for attempt in range(60):
            try:
                hf_hub_download("ARTPARK-IISc/Vaani", sh, repo_type="dataset")
                print("OK", sh, flush=True); break
            except Exception as e:
                msg = str(e)[:60]
                if "404" in msg or "EntryNotFound" in msg:
                    print("skip (not found)", sh, flush=True); break
                print(f"  {d} shard {i}: retry {attempt} ({msg})", flush=True); time.sleep(4)
print("DIALECT SHARDS DONE", flush=True)
