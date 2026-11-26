#!/bin/bash
# Sequential GPU work: both LoCoMo arms, then perplexity and MMLU-Pro for
# MemT-Nano and for its raw backbone.  One at a time -- GPUs 4 and 6 only have
# room for one copy of a 65 GiB model.
set -u
cd /mnt/ssdraid/git/deepseekv4.1
PY=/mnt/ssd/memt-env/venv/bin/python
export CUDA_VISIBLE_DEVICES=4,6 TMPDIR=/mnt/ssd/memt-env/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NANO=/mnt/ssd/models/MemT-1.0-Nano
BB=/mnt/ssd/models/Qwen3.6-35B-A3B
OUT=/mnt/ssd/memt-bench
WIKI=/mnt/ssdraid/git/deepseekv4.1/data/wikitext-2-raw/wiki.test.raw

step () { echo; echo "######## $* ########"; date +%T; }

step "LoCoMo: memory arm"
$PY memt-audit/locomo.py $NANO $OUT/locomo/locomo10.json $OUT/locomo-memory \
  --arm memory --conversations 2 --mem 36,33 > $OUT/locomo-memory.log 2>&1
tail -3 $OUT/locomo-memory.log

step "LoCoMo: full-context arm"
$PY memt-audit/locomo.py $NANO $OUT/locomo/locomo10.json $OUT/locomo-fullctx \
  --arm fullctx --conversations 2 --mem 36,33 > $OUT/locomo-fullctx.log 2>&1
tail -3 $OUT/locomo-fullctx.log

step "perplexity: MemT-Nano"
$PY memt-audit/ppl.py $NANO $WIKI $OUT/ppl-nano.json --ctx 4096 --chunks 24 --mem 36,33 \
  > $OUT/ppl-nano.log 2>&1
tail -3 $OUT/ppl-nano.log

step "perplexity: raw backbone"
$PY memt-audit/ppl.py $BB $WIKI $OUT/ppl-backbone.json --ctx 4096 --chunks 24 --mem 36,33 \
  > $OUT/ppl-backbone.log 2>&1
tail -3 $OUT/ppl-backbone.log

step "MMLU-Pro: MemT-Nano"
$PY memt-audit/mmlupro.py $NANO $OUT/mmlupro/test-00000-of-00001.parquet \
  $OUT/mmlupro-nano.json --n 280 --mem 36,33 > $OUT/mmlupro-nano.log 2>&1
tail -3 $OUT/mmlupro-nano.log

step "MMLU-Pro: raw backbone"
$PY memt-audit/mmlupro.py $BB $OUT/mmlupro/test-00000-of-00001.parquet \
  $OUT/mmlupro-backbone.json --n 280 --mem 36,33 > $OUT/mmlupro-backbone.log 2>&1
tail -3 $OUT/mmlupro-backbone.log

step "ALL DONE"
