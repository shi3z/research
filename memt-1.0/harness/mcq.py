"""MMLU-Pro by option-letter log-probability -- no generation, no parsing.

The generate-and-regex version of this measurement failed on both checkpoints
(these models write an explanation instead of a letter, and truncating it threw
away the answer).  Scoring the letter directly removes the failure mode: one
forward pass, compare log p(letter) over the ten options, argmax.  Identical
code for MemT and for its raw backbone, so the two are strictly comparable.

MemT is run with no memory committed -- the question here is only whether
attaching the memory block cost general capability.

Usage: mcq.py MODEL_DIR PARQUET OUT_JSON [--n 500] [--mem 36,33]
"""
import argparse, json, random, time, torch
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("parquet"); ap.add_argument("out_json")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--mem", default="36,33")
ap.add_argument("--attn", default="sdpa")
a = ap.parse_args()

LETTERS = "ABCDEFGHIJ"
rows = pq.read_table(a.parquet).to_pylist()
by_cat = {}
for r in rows:
    by_cat.setdefault(r["category"], []).append(r)
rng = random.Random(a.seed)
per = max(1, a.n // len(by_cat))
subset = []
for c in sorted(by_cat):
    v = by_cat[c][:]; rng.shuffle(v); subset += v[:per]
rng.shuffle(subset); subset = subset[: a.n]

caps = [int(x) for x in a.mem.split(",")]
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    a.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    attn_implementation=a.attn,
    max_memory={i: f"{caps[i]}GiB" for i in range(torch.cuda.device_count())})
model.eval()
meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
if meta:
    raise SystemExit(f"ABORT: {len(meta)} params on meta; raise --mem")
dev = next(model.parameters()).device

# Candidate token ids per letter: the letter may tokenise with or without a
# leading space depending on what precedes it, so score both and take the best.
cand = {}
for L in LETTERS:
    ids = set()
    for form in (L, " " + L):
        t = tok.encode(form, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    cand[L] = sorted(ids)
print(f"{a.model_dir}\n{len(subset)} questions | letter tokens: "
      f"{ {k: v for k, v in list(cand.items())[:3]} } ...", flush=True)
missing = [L for L in LETTERS if not cand[L]]
if missing:
    raise SystemExit(f"ABORT: no single-token form for {missing}")

recs, t0 = [], time.time()
for i, r in enumerate(subset):
    n_opt = len(r["options"])
    opts = "\n".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(r["options"]))
    user = (f"{r['question']}\n\n{opts}\n\n"
            f"Which option is correct? Reply with the letter only.")
    enc = tok.apply_chat_template([{"role": "user", "content": user}],
                                  add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True, return_dict=True, return_tensors="pt")
    ids = torch.cat([enc["input_ids"],
                     torch.tensor([tok.encode("Answer: ", add_special_tokens=False)])],
                    dim=1).to(dev)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, return_dict=True)
    logprobs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    scores = {LETTERS[j]: max(logprobs[t].item() for t in cand[LETTERS[j]])
              for j in range(n_opt)}
    pred = max(scores, key=scores.get)
    rec = {"question_id": r["question_id"], "category": r["category"],
           "gold": r["answer"], "pred": pred, "correct": int(pred == r["answer"]),
           "margin": round(scores[pred] - sorted(scores.values())[-2], 3)
                     if n_opt > 1 else None}
    recs.append(rec)
    with open(a.out_json + ".jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if i % 50 == 0:
        acc = 100 * sum(x["correct"] for x in recs) / len(recs)
        print(f"  {i+1}/{len(subset)} acc={acc:.1f}% ({(time.time()-t0)/60:.1f}m)",
              flush=True)

n = len(recs)
res = {"model": a.model_dir, "n": n, "method": "option-letter logprob argmax",
       "accuracy": round(100 * sum(x["correct"] for x in recs) / n, 2),
       "by_category": {c: round(100 * sum(x["correct"] for x in recs if x["category"] == c)
                                / max(1, sum(1 for x in recs if x["category"] == c)), 2)
                       for c in sorted({x["category"] for x in recs})},
       "minutes": round((time.time() - t0) / 60, 1)}
json.dump(res, open(a.out_json, "w"), indent=2)
print(json.dumps(res, indent=2), flush=True)
print("OK", flush=True)
