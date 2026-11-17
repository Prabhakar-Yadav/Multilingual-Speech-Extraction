#!/bin/bash
# Which OpenSLR mirror is fastest from this server? Downloads 25MB from each.
FILE=asr_bengali_0.zip
for M in \
  "https://www.openslr.org/resources/53" \
  "https://openslr.elda.org/resources/53" \
  "https://openslr.magicdatatech.com/resources/53" ; do
  echo "--- $M"
  S=$( { /usr/bin/time -f %e curl -s -r 0-26214400 -o /dev/null "$M/$FILE" ; } 2>&1 )
  echo "    25MB in ${S}s"
done
echo "--- total file size (from headers) ---"
curl -sI https://www.openslr.org/resources/53/$FILE | grep -i content-length
