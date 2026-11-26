#!/bin/bash
# Re-runs the two measurements whose harnesses were broken, after run_code.sh
# releases the GPUs: MMLU-Pro by letter-logprob, and perplexity with a chunked loss.
set -u
cd /mnt/ssdraid/git/deepseekv4.1
PY=/mnt/ssd/memt-env/venv/bin/python
export CUDA_VISIBLE_DEVICES=4,6 TMPDIR=/mnt/ssd/memt-env/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/mnt/ssd/memt-bench
NANO=/mnt/ssd/models/MemT-1.0-Nano
BB=/mnt/ssd/models/Qwen3.6-35B-A3B
WIKI=/mnt/ssdraid/git/deepseekv4.1/data/wikitext-2-raw/wiki.test.raw

while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "### coding chain released the GPUs at $(date +%T)"

for pair in "MemT-Nano $NANO nano" "backbone $BB backbone"; do
  set -- $pair
  echo "######## MMLU-Pro (logprob): $1 ########"; date +%T
  $PY memt-audit/mcq.py "$2" $OUT/mmlupro/test-00000-of-00001.parquet \
    $OUT/mcq-$3.json --n 500 --mem 36,33 > $OUT/mcq-$3.log 2>&1
  tail -4 $OUT/mcq-$3.log
done

for pair in "MemT-Nano $NANO nano" "backbone $BB backbone"; do
  set -- $pair
  echo "######## wikitext-2 perplexity: $1 ########"; date +%T
  $PY memt-audit/ppl.py "$2" $WIKI $OUT/ppl-$3.json --ctx 4096 --chunks 24 --mem 36,33 \
    > $OUT/ppl-$3.log 2>&1
  tail -4 $OUT/ppl-$3.log
done

echo "######## FIX CHAIN DONE ########"; date +%T
