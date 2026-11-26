"""Sliding-window perplexity for one model, written so MemT and its raw backbone
are measured by identical code.

"Does attaching the memory block cost general capability?" is answered here with
a judge-free, deterministic number: NLL per token on the same text, same
tokenizer, same chunking.  The paper's claim for Nano is a ~1 point drop on a
6-benchmark average; perplexity is a cheaper and more sensitive probe of the
same question.

Usage: ppl.py MODEL_DIR TEXT_FILE OUT_JSON [--ctx 4096] [--chunks 32] [--mem 34,32]
"""
import argparse, json, math, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("text_file"); ap.add_argument("out_json")
ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--chunks", type=int, default=32)
ap.add_argument("--mem", default="34,32")
ap.add_argument("--attn", default="sdpa")
a = ap.parse_args()

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

ids = tok(open(a.text_file).read(), return_tensors="pt").input_ids[0]
print(f"{a.model_dir}: {ids.numel()} tokens, ctx={a.ctx}, chunks={a.chunks}", flush=True)
dev = next(model.parameters()).device

# MemT wraps the backbone; both expose the causal-LM forward, so one path serves both.
inner = getattr(getattr(model, "model", model), "language_model", None) or model

total_nll, total_tok, per_chunk, t0 = 0.0, 0, [], time.time()
with torch.no_grad():
    for c in range(a.chunks):
        lo = c * a.ctx
        if lo + a.ctx + 1 > ids.numel():
            break
        window = ids[lo: lo + a.ctx + 1].unsqueeze(0).to(dev)
        out = model(input_ids=window[:, :-1], use_cache=False, return_dict=True)
        logits, targets = out.logits[0], window[0, 1:]
        # A 4096 x 248320 logit tensor is 2 GiB in bf16 and 4 GiB cast to fp32,
        # and cross_entropy wants another copy: cast and reduce in slices instead.
        nll = 0.0
        for s in range(0, logits.shape[0], 256):
            part = logits[s: s + 256].float()
            nll += torch.nn.functional.cross_entropy(
                part, targets[s: s + 256], reduction="sum").item()
            del part
        n = window.shape[1] - 1
        total_nll += nll; total_tok += n
        per_chunk.append(nll / n)
        if c % 4 == 0:
            print(f"  chunk {c}: ppl so far {math.exp(total_nll/total_tok):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

res = {"model": a.model_dir, "text": a.text_file, "ctx": a.ctx,
       "chunks": len(per_chunk), "tokens": total_tok,
       "ppl": round(math.exp(total_nll / total_tok), 4),
       "nll_per_token": round(total_nll / total_tok, 6),
       "per_chunk_nll": [round(x, 6) for x in per_chunk],
       "seconds": round(time.time() - t0, 1)}
json.dump(res, open(a.out_json, "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "per_chunk_nll"}, indent=2), flush=True)
print("OK", flush=True)
