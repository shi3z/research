"""Are MemT-1.0-Nano's backbone weights bit-identical to Qwen3.6-35B-A3B's?

If they are, the memory block is a pure addition and every non-memory
measurement must agree exactly -- which is what we observed.
"""
import json, sys, hashlib, torch
from safetensors import safe_open

NANO = "/mnt/ssd/models/MemT-1.0-Nano"
BB   = "/mnt/ssd/models/Qwen3.6-35B-A3B"

def index(d):
    return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]

ni, bi = index(NANO), index(BB)
print(f"nano tensors={len(ni)}  backbone tensors={len(bi)}")

# map nano names -> backbone names by stripping the wrapper prefix
def strip(n):
    for p in ("model.backbone.", "backbone.", "model.model.", "memt_model."):
        if n.startswith(p):
            return n[len(p):]
    return n

nstr = {strip(k): k for k in ni}
shared, memt_only = [], []
for k in ni:
    s = strip(k)
    if s in bi or k in bi:
        shared.append(k)
    else:
        memt_only.append(k)
print(f"shared-name tensors={len(shared)}  memt-only={len(memt_only)}")
print("memt-only sample:", memt_only[:6])
print("backbone names not in nano:", [k for k in bi if k not in ni and k not in nstr][:6])

import random
random.seed(0)
sample = shared
same = diff = 0
for k in sample:
    bk = k if k in bi else strip(k)
    with safe_open(f"{NANO}/{ni[k]}", framework="pt") as f: a = f.get_tensor(k)
    with safe_open(f"{BB}/{bi[bk]}", framework="pt") as f: b = f.get_tensor(bk)
    if a.shape != b.shape:
        print(f"SHAPE {k}: {tuple(a.shape)} vs {tuple(b.shape)}"); diff += 1; continue
    eq = torch.equal(a, b)
    if eq: same += 1
    else:
        d = (a.float() - b.float()).abs()
        print(f"DIFF  {k}: maxabs={d.max():.3e} mean={d.mean():.3e}")
        diff += 1
print(f"\nsampled {len(sample)} shared tensors: bit-identical={same}  different={diff}")
