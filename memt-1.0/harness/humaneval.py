"""HumanEval pass@1 for MemT-1.0 and its raw backbone, with identical prompting.

The paper's own general-capability table says MemT-1.0-Nano's single largest
regression is code generation: LiveCodeBench v6 -5.8 against the Qwen3.6-35B-A3B
backbone (while BigCodeBench-Hard goes +2.0).  This is a cheap, execution-based,
judge-free probe of that direction on the same two checkpoints.

HumanEval is not LiveCodeBench: it is much easier and saturated for strong
models, so treat it as a direction check, not a reproduction of -5.8.

Generated code is executed.  Each candidate runs in its own empty directory, in
a separate `python -I` process, with a wall-clock timeout and no inherited
environment.

Usage: humaneval.py MODEL_DIR PARQUET OUT_JSON [--mem 36,33] [--n 164]
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("parquet"); ap.add_argument("out_json")
ap.add_argument("--mem", default="36,33")
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--n", type=int, default=164)
ap.add_argument("--max-new-tokens", type=int, default=640)
ap.add_argument("--timeout", type=float, default=10.0)
ap.add_argument("--runner", default="/mnt/ssd/memt-env/venv/bin/python")
a = ap.parse_args()

rows = pq.read_table(a.parquet).to_pylist()[: a.n]
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
has_session = hasattr(model, "create_session")
print(f"{a.model_dir}\nsession API: {has_session}, {len(rows)} problems", flush=True)

def ask(prompt):
    if has_session:
        s = model.create_session(tok)
        out = s.chat(prompt, max_new_tokens=a.max_new_tokens, do_sample=False,
                     enable_thinking=False)
        s.close()
        return str(out)
    # transformers 5 returns a BatchEncoding here, not a bare tensor.
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt")
    ids = enc["input_ids"].to(next(model.parameters()).device)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

def build_program(row, reply):
    """Return a self-contained program: the solution plus HumanEval's own test."""
    blocks = FENCE.findall(reply)
    body = blocks[0] if blocks else reply
    if f"def {row['entry_point']}" in body:
        solution = body                       # the model restated the signature
    else:
        solution = row["prompt"] + body       # it produced only the body
    return (solution + "\n\n" + row["test"]
            + f"\n\ncheck({row['entry_point']})\n")

def run(program, workdir):
    path = os.path.join(workdir, "candidate.py")
    with open(path, "w") as fh:
        fh.write(program)
    try:
        p = subprocess.run(
            [a.runner, "-I", "candidate.py"], cwd=workdir, timeout=a.timeout,
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir,
                 "PYTHONDONTWRITEBYTECODE": "1"})
        return (p.returncode == 0), (p.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"

recs, t0 = [], time.time()
sandbox_root = tempfile.mkdtemp(prefix="humaneval-", dir="/mnt/ssd/memt-env/tmp")
for i, row in enumerate(rows):
    prompt = ("Complete the following Python function. "
              "Reply with the complete function inside one ```python code block, "
              "and nothing else.\n\n```python\n" + row["prompt"] + "```")
    try:
        reply = ask(prompt)
    except Exception as e:
        reply = f"<<ERROR {type(e).__name__}: {e}>>"
    work = os.path.join(sandbox_root, f"p{i}")
    os.makedirs(work, exist_ok=True)
    ok, err = run(build_program(row, reply), work)
    shutil.rmtree(work, ignore_errors=True)
    rec = {"task_id": row["task_id"], "passed": int(ok), "stderr": err,
           "reply_chars": len(reply), "reply": reply[:600]}
    recs.append(rec)
    with open(a.out_json + ".jsonl", "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if i % 20 == 0:
        p1 = 100 * sum(x["passed"] for x in recs) / len(recs)
        print(f"  {i+1}/{len(rows)} pass@1={p1:.1f}% ({(time.time()-t0)/60:.1f}m) "
              f"{row['task_id']} {'PASS' if ok else 'FAIL ' + err[:60]!r}", flush=True)
shutil.rmtree(sandbox_root, ignore_errors=True)

res = {"model": a.model_dir, "n": len(recs),
       "pass@1": round(100 * sum(x["passed"] for x in recs) / len(recs), 2),
       "timeouts": sum(1 for x in recs if x["stderr"] == "TIMEOUT"),
       "minutes": round((time.time() - t0) / 60, 1)}
json.dump(res, open(a.out_json, "w"), indent=2)
print(json.dumps(res, indent=2), flush=True)
print("OK", flush=True)
