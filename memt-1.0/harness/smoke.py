"""Load MemT-1.0-Nano, check it generates, and measure the real memory-state size.

Usage: smoke.py MODEL_DIR OUT_DIR
Run with CUDA_VISIBLE_DEVICES set to the two GPUs to use.
"""
import json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
n = torch.cuda.device_count()
print(f"visible GPUs: {n}", flush=True)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    # leave headroom: these GPUs are shared with other jobs
    max_memory={i: "34GiB" for i in range(n)},
)
model.eval()
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
print("device map spread:", sorted({str(p.device) for p in model.parameters()}), flush=True)

cfg = model.config
text = cfg.get_text_config()
full = [i for i, k in enumerate(text.layer_types) if k == "full_attention"]
print(f"full-attention layers: {full}", flush=True)

# --- 1. does it generate at all (memory off) ---
sess = model.create_session(tok)
t0 = time.time()
out = sess.chat("Reply with exactly: HELLO", max_new_tokens=16, do_sample=False)
print(f"plain chat ({time.time()-t0:.1f}s): {out!r}", flush=True)

# --- 2. commit a document to memory and measure the state ---
doc = ("Shigeru keeps three facts in mind. "
       "His cat is named Mochi. His favourite number is 42. He lives in Tokyo. ") * 40
sess2 = model.create_session(tok)
t0 = time.time()
out2 = sess2.chat("What is the cat's name? Answer with one word.",
                  memory=[doc], max_new_tokens=16, do_sample=False)
dt = time.time() - t0
print(f"memory chat ({dt:.1f}s): {out2!r}", flush=True)

state_path = os.path.join(out_dir, "state_smoke")
sess2.save_state(state_path, overwrite=True)
def tree_size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs)
size = tree_size(state_path)
print(f"saved state: {state_path} -> {size} bytes = {size/2**20:.2f} MiB", flush=True)
if os.path.isdir(state_path):
    for f in sorted(os.listdir(state_path)):
        print("   ", f, os.path.getsize(os.path.join(state_path, f)))

expected = len(full) * text.num_attention_heads * text.head_dim * text.head_dim * 4
print(f"predicted from config: {len(full)} layers x {text.num_attention_heads} x "
      f"{text.head_dim}^2 x 4B = {expected} bytes = {expected/2**20:.2f} MiB", flush=True)

json.dump({"state_bytes": size, "predicted_bytes": expected,
           "full_attention_layers": full, "plain": str(out), "memory": str(out2),
           "memory_chat_seconds": dt},
          open(os.path.join(out_dir, "smoke.json"), "w"), indent=2)
print("OK", flush=True)
