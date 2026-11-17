#!/bin/bash
# Independent checkpoint backup.
#
# The trainer is already configured with save_total_limit=None (keep everything),
# but we lost the best model once to a bad setting - so this is a second line of
# defence the trainer cannot undo: it copies every checkpoint it sees into
# ckpt_backup/, which nothing else writes to.
cd ~/bengali_asr
mkdir -p ckpt_backup
echo "checkpoint backup watcher started $(date)"

while true; do
  for d in ckpt_run2/checkpoint-*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    # only copy once it's fully written (trainer_state.json is written last)
    if [ -f "$d/trainer_state.json" ] && [ ! -d "ckpt_backup/$name" ]; then
      cp -r "$d" "ckpt_backup/$name.tmp" && mv "ckpt_backup/$name.tmp" "ckpt_backup/$name"
      echo "$(date +%H:%M) backed up $name"
    fi
  done

  # stop once training has exited AND we've mirrored everything
  if ! pgrep -f "python train_slr53.py" > /dev/null; then
    sleep 60
    if ! pgrep -f "python train_slr53.py" > /dev/null; then
      echo "training finished - final backup pass $(date)"
      for d in ckpt_run2/checkpoint-*; do
        [ -d "$d" ] || continue
        n=$(basename "$d")
        [ -d "ckpt_backup/$n" ] || cp -r "$d" "ckpt_backup/$n"
      done
      echo "BACKUP_WATCHER_DONE"
      break
    fi
  fi
  sleep 120
done
