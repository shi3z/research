"""LoCoMo QA on MemT-1.0, memory arm vs plain full-context arm, same backbone.

The point of this harness is the HEAD-TO-HEAD, not the absolute number.  The
paper scores LoCoMo with gpt-4.1-mini as LLM-as-a-judge using the OmniMemEval
prompt; we do not have that prompt or that judge, so absolute scores here are
NOT comparable to the paper's 57.53.  Both arms are scored by the identical
deterministic metric, so the difference between them is meaningful.

  memory arm  : each session is committed with commit_memory(document=True),
                the state is persisted, and every question is then answered in
                a fresh session restored from that state -- the question is the
                only thing in the context window.
  fullctx arm : the whole conversation is placed in the prompt, no memory used.

Usage:
  locomo.py MODEL_DIR LOCOMO_JSON OUT_DIR --arm memory|fullctx [options]

LoCoMo is CC BY-NC 4.0 (Maharana et al., ACL 2024); used here for evaluation
only and not redistributed.
"""
import argparse, json, os, re, string, sys, time, collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------- metric
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
_PUNCT = str.maketrans("", "", string.punctuation)

def norm(s):
    s = str(s).lower().replace("’", "'")
    s = _ARTICLES.sub(" ", s).translate(_PUNCT)
    return " ".join(s.split())

def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)

def recall(pred, gold):
    """Gold tokens covered by the prediction: robust to the model padding its answer."""
    p, g = collections.Counter(norm(pred).split()), collections.Counter(norm(gold).split())
    if not g:
        return 0.0
    return sum((p & g).values()) / sum(g.values())

# ----------------------------------------------------------------- data
def sessions_of(sample):
    conv = sample["conversation"]
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                  key=lambda k: int(k.split("_")[1]))
    out = []
    for k in keys:
        when = conv.get(f"{k}_date_time", "unknown time")
        lines = [f"{t['speaker']}: {t['text']}" for t in conv[k]]
        out.append(f"[{k.replace('_', ' ')}, {when}]\n" + "\n".join(lines))
    return out

INSTR = ("Answer the question using only the conversation. "
         "Reply with the shortest possible answer -- a few words, a name, or a date. "
         "Do not explain. If the conversation does not contain the answer, reply exactly: "
         "No information available.")

def gold_of(q):
    if q.get("category") == 5:
        return "No information available."
    return q.get("answer")

# ----------------------------------------------------------------- main
ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("locomo_json"); ap.add_argument("out_dir")
ap.add_argument("--arm", choices=["memory", "fullctx"], required=True)
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--mem", default="38,36")
ap.add_argument("--conversations", type=int, default=2)
ap.add_argument("--max-questions", type=int, default=0, help="0 = all")
ap.add_argument("--categories", default="1,2,3,4,5")
ap.add_argument("--max-new-tokens", type=int, default=48)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
cats = {int(x) for x in a.categories.split(",")}

data = json.load(open(a.locomo_json))[: a.conversations]
caps = [int(x) for x in a.mem.split(",")]

tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    a.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    attn_implementation=a.attn,
    max_memory={i: f"{caps[i]}GiB" for i in range(torch.cuda.device_count())})
model.eval()
meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
if meta:
    raise SystemExit(f"ABORT: {len(meta)} params on meta (disk offload); raise --mem")
print(f"arm={a.arm} attn={a.attn} conversations={len(data)}", flush=True)

records, t_start = [], time.time()
for sample in data:
    sid = sample["sample_id"]
    sess_texts = sessions_of(sample)
    convo = "\n\n".join(sess_texts)
    n_ctx = len(tok(convo).input_ids)
    qs = [q for q in sample["qa"] if q.get("category") in cats]
    if a.max_questions:
        qs = qs[: a.max_questions]
    print(f"[{sid}] {len(sess_texts)} sessions, {n_ctx} tokens, {len(qs)} questions", flush=True)

    state_path = None
    if a.arm == "memory":
        t0 = time.time()
        s = model.create_session(tok)
        s.chat("ok", memory=sess_texts, max_new_tokens=1, do_sample=False, enable_thinking=False)
        state_path = os.path.join(a.out_dir, f"state_{sid}.safetensors")
        s.save_state(state_path, overwrite=True)
        print(f"[{sid}] committed {len(sess_texts)} sessions in {time.time()-t0:.1f}s "
              f"-> {os.path.getsize(state_path)/2**20:.2f} MiB state", flush=True)
        s.close()

    for i, q in enumerate(qs):
        gold = gold_of(q)
        if gold is None:
            continue
        t0 = time.time()
        try:
            if a.arm == "memory":
                sq = model.create_session(tok, state=state_path)
                pred = sq.chat(f"{INSTR}\n\nQuestion: {q['question']}",
                               max_new_tokens=a.max_new_tokens, do_sample=False,
                               enable_thinking=False)
            else:
                sq = model.create_session(tok)
                pred = sq.chat(f"{INSTR}\n\nConversation:\n{convo}\n\nQuestion: {q['question']}",
                               max_new_tokens=a.max_new_tokens, do_sample=False,
                               enable_thinking=False)
            sq.close()
            pred = str(pred)
        except Exception as e:                      # keep going; record the failure
            pred = f"<<ERROR {type(e).__name__}: {e}>>"
        rec = {"sample_id": sid, "category": q["category"], "question": q["question"],
               "gold": str(gold), "pred": pred, "seconds": time.time() - t0,
               "f1": f1(pred, gold), "recall": recall(pred, gold),
               "em": float(norm(pred) == norm(gold)), "context_tokens": n_ctx}
        records.append(rec)
        if i % 10 == 0:
            el = time.time() - t_start
            print(f"  q{i+1}/{len(qs)} f1={rec['f1']:.2f} {rec['seconds']:.1f}s "
                  f"(elapsed {el/60:.1f}m) :: {pred[:70]!r}", flush=True)
        with open(os.path.join(a.out_dir, f"records_{a.arm}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

def agg(rs):
    if not rs:
        return None
    return {"n": len(rs),
            "f1": round(100 * sum(r["f1"] for r in rs) / len(rs), 2),
            "recall": round(100 * sum(r["recall"] for r in rs) / len(rs), 2),
            "em": round(100 * sum(r["em"] for r in rs) / len(rs), 2),
            "mean_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 2)}

summary = {"arm": a.arm, "attn": a.attn, "overall": agg(records),
           "excl_adversarial": agg([r for r in records if r["category"] != 5]),
           "by_category": {c: agg([r for r in records if r["category"] == c])
                           for c in sorted(cats)},
           "wall_minutes": round((time.time() - t_start) / 60, 1)}
json.dump(summary, open(os.path.join(a.out_dir, f"summary_{a.arm}.json"), "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)
print("OK", flush=True)
