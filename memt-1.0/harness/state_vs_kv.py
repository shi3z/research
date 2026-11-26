"""Audit MemT-1.0's storage claim from the published configs alone.

The claim (abstract, and both model cards): the persistent memory state
"remains constant in size, reducing storage by approximately 66x versus
full-context KV cache at 128K history length".

Everything below comes from config.json plus two lines of modeling_memt.py,
which fix the state tensor as (1, num_memory_heads, head_dim, head_dim) float32
per memory layer.  No model weights are needed.
"""
import json, sys

MIB = 1024 ** 2
GIB = 1024 ** 3


def state_bytes(heads, dim, layers, dtype_bytes=4):
    """modeling_memt.py: shape (1, heads, dim, dim), dtype float32, per memory layer."""
    per_layer = heads * dim * dim
    return per_layer * layers * dtype_bytes, per_layer * layers


def report(title, rows, state_b, state_params, tokens):
    print(f"\n=== {title}  (history = {tokens:,} tokens) ===")
    print(f"  memory state: {state_params/1e6:7.1f} M params fp32 = {state_b/MIB:8.1f} MiB  (constant)")
    for label, kv_b in rows:
        ratio = kv_b / state_b
        print(f"  KV {label:<34} {kv_b/GIB:7.3f} GiB   ->  {ratio:6.1f}x")


# ---------------------------------------------------------------- Nano
nano = json.load(open("/mnt/ssd/memt-probe/nano/config.json"))
t = nano["backbone_configs"]["text_config"]
m = nano["memt"]
full_attn = sum(1 for x in t["layer_types"] if x == "full_attention")
# Nano's memt block has no head_dim/layers field: modeling_memt.py line 743 uses
# the backbone's own (num_attention_heads, head_dim), one state per full-attn layer.
heads, dim = t["num_attention_heads"], t["head_dim"]
s_b, s_p = state_bytes(heads, dim, full_attn)
print(f"Nano: {t['num_hidden_layers']} layers, {full_attn} full-attention, "
      f"{t['num_key_value_heads']} kv heads x {dim} head_dim, "
      f"state {heads}x{dim}x{dim} fp32 x {full_attn}")

TOK = 128 * 1024
# KV grows only in the full-attention layers; the linear-attention layers hold a
# constant state and config says linear_attention_state_policy = reset_per_chunk.
kv_per_tok_layer = t["num_key_value_heads"] * dim * 2 * 2   # K and V, bf16
nano_rows = [
    (f"bf16, {full_attn} full-attention layers", TOK * kv_per_tok_layer * full_attn),
    (f"bf16, all {t['num_hidden_layers']} layers (hypothetical)",
     TOK * kv_per_tok_layer * t["num_hidden_layers"]),
]
report("MemT-1.0-Nano", nano_rows, s_b, s_p, TOK)

# ---------------------------------------------------------------- Flash
fl = json.load(open("/mnt/ssd/memt-probe/flash/config.json"))
fm = fl["memt"]
f_layers = len(fm["layers"])
fs_b, fs_p = state_bytes(fm["num_memory_heads"], fm["head_dim"], f_layers)
types = fl["layer_types"]
n_slide = sum(1 for x in types if x == "sliding_attention")
n_csa = sum(1 for x in types if x == "compressed_sparse_attention")
n_hca = sum(1 for x in types if x == "heavily_compressed_attention")
rate = fl["compress_rates"]
print(f"\nFlash: {fl['num_hidden_layers']} layers = {n_slide} sliding(w={fl['sliding_window']}) "
      f"+ {n_csa} csa(/{rate['compressed_sparse_attention']}) + {n_hca} hca(/{rate['heavily_compressed_attention']}), "
      f"{fl['num_key_value_heads']} kv head x {fl['head_dim']} head_dim, "
      f"state {fm['num_memory_heads']}x{fm['head_dim']}x{fm['head_dim']} fp32 x {f_layers} layers")

f_per_tok_layer = fl["num_key_value_heads"] * fl["head_dim"] * 2 * 2   # bf16
def flash_kv(dtype_bytes=2, compressed=True):
    b = dtype_bytes / 2
    per = fl["num_key_value_heads"] * fl["head_dim"] * 2 * dtype_bytes
    if not compressed:
        return TOK * per * fl["num_hidden_layers"]
    return per * (n_slide * min(TOK, fl["sliding_window"])
                  + n_csa * TOK / rate["compressed_sparse_attention"]
                  + n_hca * TOK / rate["heavily_compressed_attention"])

flash_rows = [
    ("bf16, compression reduces storage", flash_kv(2, True)),
    ("fp8_e4m3 (the launch script's setting)", flash_kv(1, True)),
    ("bf16, full KV kept, sparse read only", flash_kv(2, False)),
]
report("MemT-1.0-Flash", flash_rows, fs_b, fs_p, TOK)

print("""
Reading:
  Nano's 66x is substantially right: 64.0x by this accounting, and the
  mechanism is exactly what the card says.  The 2-3% gap is accounting, not
  a different claim.
  The figure is Nano-only and does NOT carry to Flash.  Flash's state is 32x
  larger (336M vs 10M) while its backbone KV is already compressed 4x/128x,
  so the same ratio lands between 1x and 9x depending on how the compressed
  layers store KV -- and at the fp8 KV cache the launch script actually uses,
  the persistent memory state is LARGER than the KV cache it replaces.
  Note the paper never claims 66x for Flash; the model card's placement of the
  sentence invites that reading.
""")
