"""MMLU-Pro by option-letter log-probability, for BASE (non-chat) checkpoints.

No generation and no regex: the prompt ends in "Answer:" and the answer is
argmax over log p(letter) for the options that exist.  That removes the two
failure modes of a generate+parse harness on a base model -- it ignores the
format instruction, and it gets truncated mid-sentence.

Both " A" and "A" single-token forms are scored, since the leading space is
tokenised differently.

Usage: mcq_base.py MODEL_DIR PARQUET OUT_JSON [--n 500] [--shots 0]
"""
import argparse, json, re, time, torch
import pyarrow.parquet as pq
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("parquet"); ap.add_argument("out_json")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--shots", type=int, default=0)
a = ap.parse_args()

LETTERS = "ABCDEFGHIJ"
rows = pq.read_table(a.parquet).to_pylist()
# stratify by category so a truncated run is still balanced
bycat = defaultdict(list)
for r in rows:
    bycat[r["category"]].append(r)
per = max(1, a.n // len(bycat))
sel = []
for c in sorted(bycat):
    sel.extend(bycat[c][:per])
print(f"{len(sel)} questions over {len(bycat)} categories", flush=True)

tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    a.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
model.eval()
dev = next(model.parameters()).device

def letter_ids(n_opts):
    out = []
    for i in range(n_opts):
        cands = []
        for form in (LETTERS[i], " " + LETTERS[i]):
            t = tok(form, add_special_tokens=False).input_ids
            if len(t) == 1:
                cands.append(t[0])
        out.append(cands)
    return out

def render(r):
    s = f"Question: {r['question']}\nOptions:\n"
    for i, o in enumerate(r["options"]):
        s += f"{LETTERS[i]}. {o}\n"
    return s + "Answer:"

shots = ""
if a.shots:
    pool = bycat[sorted(bycat)[0]][per: per + a.shots]
    shots = "".join(render(r) + f" {LETTERS[r['answer_index']]}\n\n" for r in pool)

recs, t0 = [], time.time()
for i, r in enumerate(sel):
    prompt = shots + render(r)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    best, best_v = None, -1e30
    for j, cands in enumerate(letter_ids(len(r["options"]))):
        if not cands:
            continue
        v = max(lp[c].item() for c in cands)
        if v > best_v:
            best_v, best = v, j
    recs.append({"category": r["category"], "gold": r["answer_index"],
                 "pred": best, "ok": int(best == r["answer_index"])})
    if i % 100 == 0:
        acc = 100 * sum(x["ok"] for x in recs) / len(recs)
        print(f"  {i+1}/{len(sel)} acc={acc:.1f}% ({(time.time()-t0)/60:.1f}m)", flush=True)

agg = defaultdict(list)
for x in recs:
    agg[x["category"]].append(x["ok"])
out = {"model": a.model_dir, "n": len(recs), "shots": a.shots,
       "method": "option-letter logprob argmax",
       "accuracy": round(100 * sum(x["ok"] for x in recs) / len(recs), 2),
       "by_category": {c: round(100 * sum(v) / len(v), 2) for c, v in sorted(agg.items())},
       "minutes": round((time.time() - t0) / 60, 1)}
json.dump(out, open(a.out_json, "w"), indent=2)
print(json.dumps(out, indent=2), flush=True)
print("OK", flush=True)
