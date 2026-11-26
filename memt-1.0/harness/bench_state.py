"""Measure MemT-1.0-Nano's persistent state size and query latency vs history length.

Tests two published claims directly:
  * "the state remains constant in size, ~66x smaller than a full-context KV
     cache at 128K history"  -> state_bytes vs history length, and the KV bytes
     the same history would need in the backbone's 10 full-attention layers.
  * "~1.86x end-to-end speedup at 128K"  -> query latency, memory arm vs
     full-context arm, at matched history lengths.

Usage: bench_state.py MODEL_DIR OUT_DIR [--attn sdpa|eager] [--max-len N]
"""
import argparse, gc, json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out_dir")
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--lengths", default="2048,8192,32768,65536,131072")
ap.add_argument("--fullctx-max", type=int, default=32768,
                help="skip the full-context arm above this many tokens")
ap.add_argument("--mem", default="38,36", help="per-GPU GiB cap")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

caps = [int(x) for x in a.mem.split(",")]
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    a.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    attn_implementation=a.attn,
    max_memory={i: f"{caps[i]}GiB" for i in range(torch.cuda.device_count())},
)
model.eval()
load_s = time.time() - t0
meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
print(f"loaded in {load_s:.1f}s attn={a.attn} | meta params: {len(meta)}", flush=True)
if meta:
    raise SystemExit(f"ABORT: {len(meta)} parameters landed on meta (disk offload). "
                     f"Raise --mem. First few: {meta[:5]}")

text = model.config.get_text_config()
full = [i for i, k in enumerate(text.layer_types) if k == "full_attention"]
kv_per_tok = text.num_key_value_heads * text.head_dim * 2 * 2 * len(full)  # bf16, K and V
print(f"{len(full)} full-attention layers -> {kv_per_tok} KV bytes/token", flush=True)

# A long, non-repetitive history: facts that a question can actually probe.
def history(n_tokens):
    parts, i = [], 0
    while True:
        i += 1
        parts.append(
            f"Log entry {i}: on day {i} the operator {['Aoi','Ren','Sora','Mio','Kai'][i%5]} "
            f"recorded reactor {i%7} at {300+(i*37)%500} kelvin with valve "
            f"{'open' if i%3 else 'shut'} and logged code {(i*7919)%100000:05d}.")
        if i % 64 == 0:
            s = " ".join(parts)
            if len(tok(s).input_ids) >= n_tokens:
                ids = tok(s).input_ids[:n_tokens]
                return tok.decode(ids), len(ids)

QUESTION = "In one short sentence: what temperature was recorded in log entry 12?"
rows = []
for n in [int(x) for x in a.lengths.split(",")]:
    doc, real = history(n)
    row = {"requested_tokens": n, "history_tokens": real,
           "kv_bytes_if_full_context": real * kv_per_tok}

    # ---- memory arm: commit the history, persist it, then query with an empty context
    sess = model.create_session(tok)
    torch.cuda.synchronize(); t0 = time.time()
    sess.chat("ok", memory=[doc], max_new_tokens=1, do_sample=False, enable_thinking=False)
    torch.cuda.synchronize(); row["commit_seconds"] = time.time() - t0

    path = os.path.join(a.out_dir, f"state_{real}.safetensors")
    sess.save_state(path, overwrite=True)
    row["state_bytes"] = os.path.getsize(path)
    row["ratio_kv_over_state"] = row["kv_bytes_if_full_context"] / row["state_bytes"]

    s2 = model.create_session(tok, state=path)
    torch.cuda.synchronize(); t0 = time.time()
    out = s2.chat(QUESTION, max_new_tokens=48, do_sample=False, enable_thinking=False)
    torch.cuda.synchronize(); row["memory_query_seconds"] = time.time() - t0
    row["memory_answer"] = str(out)[:200]

    # ---- full-context arm: the same history in the prompt, no memory at all
    if real <= a.fullctx_max:
        s3 = model.create_session(tok)
        torch.cuda.synchronize(); t0 = time.time()
        try:
            out3 = s3.chat(f"{doc}\n\n{QUESTION}", max_new_tokens=48,
                           do_sample=False, enable_thinking=False)
            torch.cuda.synchronize()
            row["fullctx_query_seconds"] = time.time() - t0
            row["fullctx_answer"] = str(out3)[:200]
            row["speedup_memory_vs_fullctx"] = (
                row["fullctx_query_seconds"] / row["memory_query_seconds"])
        except torch.cuda.OutOfMemoryError:
            row["fullctx_query_seconds"] = None; row["fullctx_answer"] = "OOM"
    else:
        row["fullctx_query_seconds"] = None
        row["fullctx_answer"] = f"skipped (> --fullctx-max {a.fullctx_max})"

    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    gc.collect(); torch.cuda.empty_cache()

json.dump({"attn": a.attn, "load_seconds": load_s, "full_attention_layers": full,
           "kv_bytes_per_token": kv_per_tok, "rows": rows},
          open(os.path.join(a.out_dir, "bench_state.json"), "w"), indent=2)
print("\n  history    state      KV if full-ctx   ratio   commit    mem-q   full-q  speedup")
for r in rows:
    f = r.get("fullctx_query_seconds")
    print(f"  {r['history_tokens']:>7}  {r['state_bytes']/2**20:6.2f}MiB  "
          f"{r['kv_bytes_if_full_context']/2**20:9.1f}MiB  {r['ratio_kv_over_state']:6.1f}x  "
          f"{r['commit_seconds']:7.1f}s {r['memory_query_seconds']:7.2f}s  "
          f"{(f'{f:6.2f}s' if f else '     --')}  "
          f"{(f'{r[chr(115)+chr(112)+chr(101)+chr(101)+chr(100)+chr(117)+chr(112)+chr(95)+chr(109)+chr(101)+chr(109)+chr(111)+chr(114)+chr(121)+chr(95)+chr(118)+chr(115)+chr(95)+chr(102)+chr(117)+chr(108)+chr(108)+chr(99)+chr(116)+chr(120)]:5.2f}x' if f else '   --')}")
print("OK", flush=True)
