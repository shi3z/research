"""Needle-in-a-haystack for a model with NO global attention.

SpikingBrain-7B is 14 sliding-window(4096) layers interleaved with 14 GLA
(gated linear attention) layers.  Every layer is local: the SWA layers see at
most 4096 tokens, the GLA layers carry a fixed-size recurrent state.  Nothing
in the stack can attend to an arbitrary earlier token exactly.

The headline claim is prefill of 4M-token sequences.  Processing 4M tokens
cheaply is easy if you never attend globally; this asks whether the information
survives.  Qwen2.5-7B (the checkpoint SpikingBrain was converted from, full
attention, 131k positions) is the control: same tokenizer, same dimensions, so
the only difference is the attention.

Scoring is exact-match on the retrieved number, which needs no judge.

Usage: needle.py MODEL_DIR OUT_JSON [--lengths 2048,8192,32768] [--depths 0.1,0.5,0.9]
"""
import argparse, json, random, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out_json")
ap.add_argument("--lengths", default="2048,4096,8192,16384,32768")
ap.add_argument("--depths", default="0.1,0.25,0.5,0.75,0.9")
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--mem", default="70")
ap.add_argument("--max-new-tokens", type=int, default=24)
a = ap.parse_args()

LENGTHS = [int(x) for x in a.lengths.split(",")]
DEPTHS = [float(x) for x in a.depths.split(",")]

tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
caps = [int(x) for x in a.mem.split(",")]
model = AutoModelForCausalLM.from_pretrained(
    a.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={i: f"{caps[min(i,len(caps)-1)]}GiB" for i in range(torch.cuda.device_count())})
model.eval()
meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
if meta:
    raise SystemExit(f"ABORT: {len(meta)} params on meta; raise --mem")
dev = next(model.parameters()).device
print(f"{a.model_dir}\n{sum(p.numel() for p in model.parameters())/1e9:.3f} G params on {dev}", flush=True)

# Filler that carries no numbers, so the only digits in the context are the needle's.
FILLER = ("The grass was wet with morning dew and the air smelled of rain. "
          "Somewhere a door closed softly. The hills kept their own counsel. "
          "A train passed behind the trees without slowing. ")
filler_ids = tok(FILLER, add_special_tokens=False).input_ids

def build(length, depth, secret):
    needle = f" The secret access code for the vault is {secret}. "
    nid = tok(needle, add_special_tokens=False).input_ids
    q = tok("\n\nQuestion: What is the secret access code for the vault? "
            "Answer with the number only.\nAnswer:", add_special_tokens=False).input_ids
    budget = length - len(nid) - len(q)
    if budget < 64:
        return None
    reps = budget // len(filler_ids) + 1
    hay = (filler_ids * reps)[:budget]
    cut = int(len(hay) * depth)
    return hay[:cut] + nid + hay[cut:] + q

rng = random.Random(0)
recs = []
for L in LENGTHS:
    for d in DEPTHS:
        hits = 0
        for t in range(a.trials):
            secret = str(rng.randint(10000, 99999))
            ids = build(L, d, secret)
            if ids is None:
                continue
            x = torch.tensor([ids], device=dev)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(x, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            reply = tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)
            ok = secret in reply
            hits += ok
            recs.append({"length": L, "depth": d, "trial": t, "secret": secret,
                         "tokens": int(x.shape[1]), "hit": int(ok),
                         "reply": reply[:120].replace("\n", " "),
                         "seconds": round(time.time() - t0, 2)})
            with open(a.out_json + ".jsonl", "a") as fh:
                fh.write(json.dumps(recs[-1], ensure_ascii=False) + "\n")
        print(f"  len={L:>6} depth={d:<5} hit={hits}/{a.trials}", flush=True)
    torch.cuda.empty_cache()

grid = {}
for r in recs:
    grid.setdefault(r["length"], {}).setdefault(r["depth"], []).append(r["hit"])
summary = {"model": a.model_dir,
           "by_length": {str(L): round(100 * sum(sum(v) for v in d.values())
                                       / max(1, sum(len(v) for v in d.values())), 1)
                         for L, d in grid.items()},
           "grid": {str(L): {str(dp): round(100 * sum(v) / len(v), 1) for dp, v in d.items()}
                    for L, d in grid.items()},
           "n": len(recs)}
json.dump(summary, open(a.out_json, "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)
print("OK", flush=True)
