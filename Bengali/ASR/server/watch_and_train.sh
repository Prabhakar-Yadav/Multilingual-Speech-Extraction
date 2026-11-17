#!/bin/bash
# Waits until enough GPU memory is free, then auto-launches the large-v3 training.
# So you don't have to babysit the other user's job - the moment it frees up,
# training starts on its own. Runs under nohup, survives disconnect.
cd ~/bengali_asr

NEED_MB=15000      # large-v3 needs ~14GB; wait for 15GB free to be safe
echo "watcher started $(date). waiting for ${NEED_MB}MB free..."

while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "$(date +%H:%M): free ${FREE} MB"
  if [ "${FREE:-0}" -ge "$NEED_MB" ]; then
    echo ">>> enough free ($FREE MB) - LAUNCHING large-v3 training"
    ./venv/bin/python train_slr53.py > train_run2.log 2>&1
    echo ">>> training process exited $(date)"
    break
  fi
  sleep 120     # re-check every 2 minutes
done
