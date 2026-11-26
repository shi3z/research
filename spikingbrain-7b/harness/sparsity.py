"""How sparse is the "spiking" activation, measured with the authors' own encoder?

The claim is "69.15% sparsity" (report, abstract).  The quantity is produced by
W8ASpike/quant_linear.py::dynamic_spikes, reproduced verbatim below:

    vth = x.abs().mean([-1], keepdim=True) / k      # k = 3.0
    spikes_int = (x / vth).round()

so a value is silent exactly when |x| < vth/2 = mean|x|/(2k).

Two different quantities both get called "sparsity", and they are not the same:

  (a) ZERO FRACTION -- how often a neuron emits nothing, i.e. spikes_int == 0.
      This is what "69% sparsity" reads as, and what an event-driven chip would
      actually skip.

  (b) The report's own number.  Section A.8: "94.1% of values are within 7
      (i.e. within the INT3 range), we further analyze the firing rate under a
      step size of 3.  By filling inactive bits within the 3-step window with 0,
      the overall sparsity reaches about 69.15%."  So each spike count is
      expanded into a 3-bit/3-timestep window and the ZERO BITS of that
      expansion are counted -- bit-level, not neuron-level.  It follows almost
      directly from the report's stated average spike count of 1.13: about 0.93
      set bits out of 3 is 69% zeros.

This measures both, plus the mean |spike| so (b) can be checked against 1.13.

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

# name -> [zeros, total values, sum|spike|, set bits in the 3-step expansion, within-INT3 count]
stats = defaultdict(lambda: [0, 0, 0.0, 0, 0])
def hook(name):
    def f(mod, inp, out):
        x = inp[0].detach().float()
        s = dynamic_spikes(x, a.k)
        mag = s.abs()
        stats[name][0] += (s == 0).sum().item()
        stats[name][1] += s.numel()
        stats[name][2] += mag.sum().item()
        # the report's quantity: |v| clamped into INT3 and written as 3 bits,
        # inactive bits of the 3-step window filled with 0 -> count the 1 bits
        c = mag.clamp(max=7).to(torch.int32)
        bits = (c & 1) + ((c >> 1) & 1) + ((c >> 2) & 1)
        stats[name][3] += bits.sum().item()
        stats[name][4] += (mag <= 7).sum().item()
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
tot_b = sum(v[3] for v in stats.values())
tot_i = sum(v[4] for v in stats.values())
T = 3

def group(n):
    return re.sub(r"\.layers\.\d+\.", ".layers.N.", n)
agg = defaultdict(lambda: [0, 0, 0.0, 0, 0])
for n, v in stats.items():
    g = agg[group(n)]
    for i in range(5):
        g[i] += v[i]

print(f"\n{'linear group':<42} {'zero %':>8} {'bit-0 %':>9} {'mean|spk|':>10}")
for g in sorted(agg):
    z, n, sm, b, _ = agg[g]
    print(f"{g:<42} {100*z/n:>8.2f} {100*(1-b/(T*n)):>9.2f} {sm/n:>10.3f}")
print(f"\n{'OVERALL':<42} {100*tot_z/tot_n:>8.2f} {100*(1-tot_b/(T*tot_n)):>9.2f} {tot_a/tot_n:>10.3f}")
print(f"\n  (a) zero fraction of spike counts : {100*tot_z/tot_n:.2f} %")
print(f"  (b) report's 3-step bit sparsity  : {100*(1-tot_b/(T*tot_n)):.2f} %   (report: 69.15 %)")
print(f"      mean |spike count|            : {tot_a/tot_n:.3f}       (report: 1.13)")
print(f"      within INT3 range (|v| <= 7)  : {100*tot_i/tot_n:.2f} %   (report: 94.1 %)")

out = {"model": a.model_dir, "k": a.k, "tokens": int(ids.shape[1]),
       "zero_fraction_pct": round(100 * tot_z / tot_n, 2),
       "report_3step_bit_sparsity_pct": round(100 * (1 - tot_b / (T * tot_n)), 2),
       "report_claim_pct": 69.15,
       "mean_abs_spike": round(tot_a / tot_n, 4),
       "report_mean_spike_claim": 1.13,
       "within_int3_pct": round(100 * tot_i / tot_n, 2),
       "report_within_int3_claim": 94.1,
       "n_linear_hooked": len(handles),
       "by_group": {g: {"zero_pct": round(100 * v[0] / v[1], 2),
                        "bit_sparsity_pct": round(100 * (1 - v[3] / (T * v[1])), 2),
                        "mean_abs_spike": round(v[2] / v[1], 4)}
                    for g, v in agg.items()}}
json.dump(out, open(a.out_json, "w"), indent=2)
print("OK", flush=True)
