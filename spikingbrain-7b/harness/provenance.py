"""Is SpikingBrain-7B initialised from Qwen2.5-7B, or trained from scratch?

The two configs are identical dimension for dimension (hidden 3584, 28 layers,
intermediate 18944, 28/4 heads, vocab 152064, rope_theta 1e6), which is already
suggestive.  This compares the weights that the GLA conversion does NOT touch --
the embedding, the output head, and every MLP -- against Qwen2.5-7B.

Reading the SpikingBrain shards means unpickling a .bin, so every load here is
weights_only=True: tensors only, no arbitrary code.

Usage: provenance.py SPIKINGBRAIN_DIR QWEN_DIR [--per-layer N]
"""
import argparse, json, os, re, sys, torch
from collections import defaultdict
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("sb_dir"); ap.add_argument("qwen_dir")
ap.add_argument("--per-layer", type=int, default=28)
a = ap.parse_args()

# ---- Qwen2.5-7B: safetensors, lazy per-tensor reads
qidx = json.load(open(f"{a.qwen_dir}/model.safetensors.index.json"))["weight_map"]
def qwen(name):
    with safe_open(f"{a.qwen_dir}/{qidx[name]}", framework="pt") as f:
        return f.get_tensor(name)

# ---- SpikingBrain: sharded .bin; load one shard at a time, tensors only
sidx = json.load(open(f"{a.sb_dir}/pytorch_model.bin.index.json"))["weight_map"]
by_shard = defaultdict(list)
for n, s in sidx.items():
    by_shard[s].append(n)
print(f"spikingbrain tensors={len(sidx)} shards={len(by_shard)}")
print(f"qwen2.5 tensors={len(qidx)}")

# SpikingBrain names: model.layers.N.{attn,mlp}.*  /  Qwen: model.layers.N.{self_attn,mlp}.*
def to_qwen(n):
    n = n.replace(".attn.", ".self_attn.")
    return n

rows, missing = [], []
for shard in sorted(by_shard):
    sd = torch.load(f"{a.sb_dir}/{shard}", map_location="cpu", weights_only=True)
    for n in sorted(by_shard[shard]):
        qn = to_qwen(n)
        if qn not in qidx:
            missing.append(n); continue
        t = sd[n]
        q = qwen(qn)
        if t.shape != q.shape:
            rows.append((n, tuple(t.shape), tuple(q.shape), None, None, None)); continue
        tf, qf = t.float(), q.float()
        identical = torch.equal(t, q)
        d = (tf - qf)
        rel = (d.norm() / qf.norm().clamp(min=1e-12)).item()
        # cosine over the flattened tensor: survives any global rescale
        cos = torch.nn.functional.cosine_similarity(tf.flatten(), qf.flatten(), dim=0).item()
        rows.append((n, tuple(t.shape), tuple(q.shape), identical, rel, cos))
    del sd

print(f"\nSpikingBrain-only tensors (no Qwen counterpart): {len(missing)}")
for n in missing[:12]: print("   ", n)
if len(missing) > 12: print(f"    ... and {len(missing)-12} more")

ident = [r for r in rows if r[3] is True]
close = [r for r in rows if r[3] is False and r[5] is not None and r[5] > 0.99]
print(f"\nshared-name tensors compared: {len(rows)}")
print(f"  bit-identical to Qwen2.5-7B : {len(ident)}")
print(f"  cosine > 0.99 (not identical): {len(close)}")

def group(n):
    return re.sub(r"\.layers\.\d+\.", ".layers.N.", n)
agg = defaultdict(list)
for n, ts, qs, idt, rel, cos in rows:
    if cos is not None:
        agg[group(n)].append((idt, rel, cos))
print(f"\n{'tensor group':<46} {'n':>3} {'identical':>9} {'median rel':>11} {'median cos':>11}")
for g in sorted(agg):
    v = agg[g]
    ni = sum(1 for x in v if x[0])
    rel = sorted(x[1] for x in v)[len(v)//2]
    cos = sorted(x[2] for x in v)[len(v)//2]
    print(f"{g:<46} {len(v):>3} {ni:>9} {rel:>11.4f} {cos:>11.6f}")

out = {"n_compared": len(rows), "n_bit_identical": len(ident),
       "n_cosine_gt_099": len(close), "n_spikingbrain_only": len(missing),
       "groups": {g: {"n": len(v), "identical": sum(1 for x in v if x[0]),
                      "median_rel_err": sorted(x[1] for x in v)[len(v)//2],
                      "median_cosine": sorted(x[2] for x in v)[len(v)//2]}
                  for g, v in agg.items()},
       "spikingbrain_only": missing}
json.dump(out, open("provenance.json", "w"), indent=2)
print("\nwrote provenance.json")
