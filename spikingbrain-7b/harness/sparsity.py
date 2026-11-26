"""How sparse is the "spiking" activation, measured with the authors' own encoder?

The claim is "69.15% sparsity" (report, abstract).  The quantity is produced by
W8ASpike/quant_linear.py::dynamic_spikes, reproduced verbatim below:

    vth = x.abs().mean([-1], keepdim=True) / k      # k = 3.0
    spikes_int = (x / vth).round()

so a value is silent exactly when |x| < vth/2 = mean|x|/(2k).  This hooks every
nn.Linear in the model, applies that function to the real input activations, and
counts zeros.

Note what the shipped forward() then does with the spikes:

    x = (spikes_int * vth).to(x.dtype)
    out = torch.nn.functional.linear(x, weight, self.bias)

It multiplies them straight back by the threshold and calls a DENSE matmul.  The
sparse-addition path (`spike_matmul`) is present but commented out.  So the
sparsity measured here is real, and as shipped it buys nothing at runtime.

Usage: sparsity.py MODEL_DIR TEXT_FILE OUT_JSON [--k 3.0] [--tokens 4096]
"""
import argparse, json, re, torch, torch.nn as nn
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("text_file"); ap.add_argument("out_json")
ap.add_argument("--k", type=float, default=3.0)
ap.add_argument("--tokens", type=int, default=4096)
a = ap.parse_args()

def dynamic_spikes(x, k):
    """Verbatim from W8ASpike/quant_linear.py (minus the optional Int2Spike call)."""
    vth = x.abs().mean([-1], keepdim=True).float() / k
    vth = vth.clamp(min=1e-5, max=1e4)
    return (x / vth).round()

tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    a.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
model.eval()

stats = defaultdict(lambda: [0, 0, 0.0])   # name -> [zeros, total, sum|spike|]
def hook(name):
    def f(mod, inp, out):
        x = inp[0].detach().float()
        s = dynamic_spikes(x, a.k)
        z = (s == 0).sum().item()
        stats[name][0] += z
        stats[name][1] += s.numel()
        stats[name][2] += s.abs().sum().item()
    return f

handles = []
for n, m in model.named_modules():
    if isinstance(m, nn.Linear) and "lm_head" not in n:
        handles.append(m.register_forward_hook(hook(n)))
print(f"hooked {len(handles)} nn.Linear modules", flush=True)

ids = tok(open(a.text_file).read(), return_tensors="pt").input_ids[:, : a.tokens].to(0)
print(f"forward over {ids.shape[1]} tokens", flush=True)
with torch.no_grad():
    model(input_ids=ids, use_cache=False)
for h in handles:
    h.remove()

tot_z = sum(v[0] for v in stats.values())
tot_n = sum(v[1] for v in stats.values())
tot_a = sum(v[2] for v in stats.values())

def group(n):
    return re.sub(r"\.layers\.\d+\.", ".layers.N.", n)
agg = defaultdict(lambda: [0, 0, 0.0])
for n, v in stats.items():
    g = agg[group(n)]
    g[0] += v[0]; g[1] += v[1]; g[2] += v[2]

print(f"\n{'linear group':<46} {'sparsity %':>11} {'mean |spike|':>13}")
for g in sorted(agg):
    z, n, s = agg[g]
    print(f"{g:<46} {100*z/n:>11.2f} {s/n:>13.3f}")
print(f"\n{'OVERALL':<46} {100*tot_z/tot_n:>11.2f} {tot_a/tot_n:>13.3f}")

out = {"model": a.model_dir, "k": a.k, "tokens": int(ids.shape[1]),
       "overall_sparsity_pct": round(100 * tot_z / tot_n, 2),
       "overall_mean_abs_spike": round(tot_a / tot_n, 4),
       "n_linear_hooked": len(handles),
       "by_group": {g: {"sparsity_pct": round(100 * v[0] / v[1], 2),
                        "mean_abs_spike": round(v[2] / v[1], 4)}
                    for g, v in agg.items()}}
json.dump(out, open(a.out_json, "w"), indent=2)
print("OK", flush=True)
