"""Download more Vaani Kolkata shards (each ~487MB, ~109 transcribed Bengali clips)
into the HF cache, with per-shard retries for a flaky link. prep_data.py auto-picks
them up. Run in background; safe to re-run (resumes).

    .\.venv-train\Scripts\python.exe -u Bengali_ASR\get_shards.py
"""
import time
from huggingface_hub import hf_hub_download

for i in range(1, 7):                      # shards 1..6 (shard 0 already downloaded)
    sh = f"audio/WestBengal/Kolkata/train-{i:05d}-of-00047.parquet"
    for attempt in range(80):
        try:
            hf_hub_download("ARTPARK-IISc/Vaani", sh, repo_type="dataset")
            print("OK", sh, flush=True); break
        except Exception as e:
            print(f"  {sh}: retry {attempt} ({str(e)[:50]})", flush=True); time.sleep(4)
print("SHARDS DONE", flush=True)
