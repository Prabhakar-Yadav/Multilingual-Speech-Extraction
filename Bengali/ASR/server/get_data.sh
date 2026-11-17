#!/bin/bash
# Download OpenSLR SLR53 (clean READ Bengali). ~920MB per part.
# Downloads parts IN PARALLEL - a single stream gets throttled (~0.4MB/s),
# parallel streams reach the server's real speed (~4MB/s+).
# Safe to re-run: -c resumes partial files, unzip -n skips existing.
set -u
cd ~/bengali_asr

N_PARTS=${N_PARTS:-6}
BASE=https://www.openslr.org/resources/53

wget -q -nc $BASE/utt_spk_text.tsv

echo "downloading $N_PARTS parts in parallel..."
for i in $(seq 0 $((N_PARTS-1))); do
  ( wget -q -c -O asr_bengali_$i.zip $BASE/asr_bengali_$i.zip && echo "  part $i downloaded" ) &
done
wait
echo "all downloads finished - extracting..."

for i in $(seq 0 $((N_PARTS-1))); do
  unzip -q -n asr_bengali_$i.zip -d slr53 && echo "  part $i extracted"
done

echo "TOTAL CLIPS:"
find slr53 -name '*.flac' | wc -l
echo "DATA_DONE"
