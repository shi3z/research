"""Prefill (time-to-first-token) scaling, SpikingBrain-7B vs its Qwen2.5-7B parent.

The paper measures 26.5x at 1M tokens on H100 WITH multi-device sequence
parallelism (ZeCO + p2p), and extrapolates ">100x at 4M" from a fitted curve --
it says so: "Due to resource limitations ... Qwen2.5 is difficult to evaluate at
4M tokens. Based on the fitted scaling curve, we conservatively estimate a
speedup of over 100x."  Its own single-device numbers are 1.41x (HuggingFace)
and 2.75x (vLLM) at 128k.

This measures the single-GPU regime on one A100 80GB: the lengths a user
actually runs.  Timing is of the transformer body only (model.model), because
applying the LM head to every position would allocate L x 152064 logits and
dominate; TTFT in practice applies the head to one position.

Usage: ttft.py MODEL_DIR OUT_JSON [--lengths 4096,...] [--reps 3]
"""
import argparse, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out_json")
ap.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--warmup", type=int, default=1)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    a.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
model.eval()
body = model.model          # transformer body, no LM head
dev = next(model.parameters()).device
print(f"{a.model_dir} on {dev}", flush=True)

rows = []
for L in [int(x) for x in a.lengths.split(",")]:
    ids = torch.randint(1000, 100000, (1, L), device=dev)
    try:
        for _ in range(a.warmup):
            with torch.no_grad():
                body(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.reps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                body(input_ids=ids, use_cache=False)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        peak = torch.cuda.max_memory_allocated() / 2**30
        ts.sort()
        rows.append({"length": L, "seconds": round(ts[len(ts)//2], 4),
                     "min": round(ts[0], 4), "max": round(ts[-1], 4),
                     "tok_per_s": round(L / ts[len(ts)//2]), "peak_gib": round(peak, 2)})
        print(f"  {L:>7} tok: {ts[len(ts)//2]:7.3f} s  ({L/ts[len(ts)//2]:>8.0f} tok/s)  peak {peak:.1f} GiB", flush=True)
    except torch.OutOfMemoryError:
        rows.append({"length": L, "oom": True})
        print(f"  {L:>7} tok: OOM", flush=True)
        torch.cuda.empty_cache()
        break
    del ids
    torch.cuda.empty_cache()

json.dump({"model": a.model_dir, "rows": rows}, open(a.out_json, "w"), indent=2)
print("OK", flush=True)
