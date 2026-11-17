#!/bin/bash
# Stop training cleanly at a given wall-clock time.   usage: ./stop_at.sh 21:30
#
# Sends SIGINT (not SIGKILL) so HuggingFace Trainer unwinds normally instead of
# being torn down mid-write, which could corrupt the checkpoint being saved.
TARGET="${1:-21:30}"
cd ~/bengali_asr

PID=$(pgrep -f "[t]rain_slr53.py" | head -1)
if [ -z "$PID" ]; then echo "no training process found"; exit 1; fi

SECS=$(( $(date -d "$TARGET" +%s) - $(date +%s) ))
if [ "$SECS" -le 0 ]; then echo "$TARGET is in the past"; exit 1; fi

echo "training pid=$PID -> will stop at $TARGET (in $((SECS/60)) min)"
sleep "$SECS"

kill -INT "$PID" 2>/dev/null
sleep 30
kill -0 "$PID" 2>/dev/null && kill -TERM "$PID" 2>/dev/null   # only if it ignored SIGINT

echo "stopped at $(date '+%H:%M')"
echo "checkpoints on disk:"
ls -1 ckpt_run2/ 2>/dev/null
