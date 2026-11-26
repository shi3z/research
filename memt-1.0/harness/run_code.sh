#!/bin/bash
# Runs after run_all.sh releases the GPUs: execution-based coding comparison.
set -u
cd /mnt/ssdraid/git/deepseekv4.1
PY=/mnt/ssd/memt-env/venv/bin/python
export CUDA_VISIBLE_DEVICES=4,6 TMPDIR=/mnt/ssd/memt-env/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/mnt/ssd/memt-bench
NANO=/mnt/ssd/models/MemT-1.0-Nano
BB=/mnt/ssd/models/Qwen3.6-35B-A3B

while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "### prior chain released the GPUs at $(date +%T)"

echo "######## HumanEval: MemT-Nano ########"; date +%T
$PY memt-audit/humaneval.py $NANO $OUT/humaneval/test.parquet \
  $OUT/humaneval-nano.json --mem 36,33 > $OUT/humaneval-nano.log 2>&1
tail -8 $OUT/humaneval-nano.log

echo "######## HumanEval: raw backbone ########"; date +%T
$PY memt-audit/humaneval.py $BB $OUT/humaneval/test.parquet \
  $OUT/humaneval-backbone.json --mem 36,33 > $OUT/humaneval-backbone.log 2>&1
tail -8 $OUT/humaneval-backbone.log

echo "######## CODE DONE ########"; date +%T
