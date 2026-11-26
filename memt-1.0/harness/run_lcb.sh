#!/bin/bash
# LiveCodeBench v6: the benchmark the paper says Nano regresses 5.8 points on.
set -u
cd /mnt/ssdraid/git/deepseekv4.1
PY=/mnt/ssd/memt-env/venv/bin/python
export CUDA_VISIBLE_DEVICES=4,6 TMPDIR=/mnt/ssd/memt-env/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/mnt/ssd/memt-bench

while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "### prior chain released the GPUs at $(date +%T)"

for pair in "MemT-Nano /mnt/ssd/models/MemT-1.0-Nano nano" \
            "backbone /mnt/ssd/models/Qwen3.6-35B-A3B backbone"; do
  set -- $pair
  echo "######## LiveCodeBench v6: $1 ########"; date +%T
  $PY memt-audit/lcb.py "$2" $OUT/lcb/test6.jsonl $OUT/lcb-$3.json --mem 36,33 \
    > $OUT/lcb-$3.log 2>&1
  tail -6 $OUT/lcb-$3.log
done
echo "######## LCB DONE ########"; date +%T
