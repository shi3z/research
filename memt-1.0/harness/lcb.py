"""LiveCodeBench v6 pass@1 for MemT-1.0 and its raw backbone, same prompting.

This is the benchmark the paper's own table says MemT-1.0-Nano regresses on
hardest: LiveCodeBench v6 -5.8 against Qwen3.6-35B-A3B (its largest regression
anywhere in that table, while BigCodeBench-Hard goes +2.0).  So measure that
benchmark on those two checkpoints.

IMPORTANT about absolute numbers: this runs with thinking DISABLED and a
1536-token budget, because thinking-on generation for 175 problems x 2 models
does not fit the time available.  The paper's 88.06 for the backbone is
certainly a thinking-on number.  Scores here will therefore be far lower than
the paper's; the measurement is the Nano-minus-backbone DELTA under identical
settings, not a reproduction of 88.06.

Generated code is executed: one `python -I` subprocess per test (stdin problems)
or per problem (functional problems), each in its own empty directory, with
wall-clock timeouts and no inherited environment.

Usage: lcb.py MODEL_DIR TEST6_JSONL OUT_JSON [--mem 36,33] [--n 175]
"""
import argparse, base64, json, os, pickle, re, shutil, subprocess, tempfile, time, zlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("jsonl"); ap.add_argument("out_json")
ap.add_argument("--mem", default="36,33")
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--n", type=int, default=175)
ap.add_argument("--max-new-tokens", type=int, default=1536)
ap.add_argument("--thinking", action="store_true", help="leave thinking enabled")
ap.add_argument("--test-timeout", type=float, default=6.0)
ap.add_argument("--problem-budget", type=float, default=90.0)
ap.add_argument("--runner", default="/mnt/ssd/memt-env/venv/bin/python")
a = ap.parse_args()

def decode(s):
    try:
        v = json.loads(s)
    except Exception:
        raw = zlib.decompress(base64.b64decode(s.encode()))
        try: v = json.loads(raw)
        except Exception: v = pickle.loads(raw)
    return json.loads(v) if isinstance(v, str) else v

problems = [json.loads(l) for l in open(a.jsonl)][: a.n]
for p in problems:
    p["tests"] = decode(p["public_test_cases"]) + decode(p["private_test_cases"])

caps = [int(x) for x in a.mem.split(",")]
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    a.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    attn_implementation=a.attn,
    max_memory={i: f"{caps[i]}GiB" for i in range(torch.cuda.device_count())})
model.eval()
meta = [n for n, q in model.named_parameters() if q.device.type == "meta"]
if meta:
    raise SystemExit(f"ABORT: {len(meta)} params on meta; raise --mem")
has_session = hasattr(model, "create_session")
think = True if a.thinking else False
print(f"{a.model_dir}\nsession={has_session} thinking={think} "
      f"problems={len(problems)} budget={a.max_new_tokens} tokens", flush=True)

def ask(prompt):
    if has_session:
        s = model.create_session(tok)
        out = s.chat(prompt, max_new_tokens=a.max_new_tokens, do_sample=False,
                     enable_thinking=think)
        s.close()
        return str(out)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        enable_thinking=think, tokenize=True, return_dict=True, return_tensors="pt")
    ids = enc["input_ids"].to(next(model.parameters()).device)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

# LiveCodeBench's own two prompt shapes.
STDIN = ("You will be given a question (problem specification) and will generate a "
         "correct Python program that matches the specification and passes all tests. "
         "The program must read from standard input and write to standard output.\n\n"
         "### Question\n{q}\n\n"
         "Put your final solution in a single ```python code block.")
FUNC = ("You will be given a question (problem specification) and will generate a "
        "correct Python program that matches the specification and passes all tests.\n\n"
        "### Question\n{q}\n\n### Starter code\n```python\n{s}\n```\n\n"
        "Complete the solution using the starter code, in a single ```python code block.")
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

def extract(reply):
    blocks = FENCE.findall(reply)
    return blocks[-1] if blocks else reply

DRIVER = '''import json, sys
{code}

def norm(v):
    if isinstance(v, tuple): return [norm(x) for x in v]
    if isinstance(v, list):  return [norm(x) for x in v]
    return v

tests = json.load(open("tests.json"))
sol = Solution()
name = [m for m in dir(sol) if not m.startswith("_")][0]
fn = getattr(sol, name)
for t in tests:
    args = [json.loads(x) for x in t["input"].rstrip("\\n").split("\\n")]
    got = norm(fn(*args))
    want = norm(json.loads(t["output"]))
    if got != want:
        print("MISMATCH", json.dumps(got)[:200], "!=", json.dumps(want)[:200])
        sys.exit(1)
print("ALL OK")
'''

def judge(problem, code, work):
    tests = problem["tests"]
    kind = tests[0]["testtype"]
    t0 = time.time()
    if kind == "functional":
        starter = problem["starter_code"]
        body = code if "class Solution" in code else starter + "\n" + code
        prelude = "from typing import List, Optional, Dict, Tuple, Any\n" \
                  "from collections import *\nfrom math import *\nimport heapq, bisect, itertools, functools, re\n"
        with open(os.path.join(work, "tests.json"), "w") as fh:
            json.dump(tests, fh)
        with open(os.path.join(work, "run.py"), "w") as fh:
            fh.write(DRIVER.format(code=prelude + body))
        try:
            p = subprocess.run([a.runner, "-I", "run.py"], cwd=work,
                               timeout=a.problem_budget, capture_output=True, text=True,
                               env={"PATH": "/usr/bin:/bin", "HOME": work, "TMPDIR": work})
            return p.returncode == 0, (p.stdout + p.stderr)[-200:]
        except subprocess.TimeoutExpired:
            return False, "PROBLEM_TIMEOUT"
    # stdin: the program consumes stdin and exits, so one process per test
    with open(os.path.join(work, "cand.py"), "w") as fh:
        fh.write(code)
    for i, t in enumerate(tests):
        if time.time() - t0 > a.problem_budget:
            return False, f"PROBLEM_TIMEOUT after {i} tests"
        try:
            p = subprocess.run([a.runner, "-I", "cand.py"], cwd=work, input=t["input"],
                               timeout=a.test_timeout, capture_output=True, text=True,
                               env={"PATH": "/usr/bin:/bin", "HOME": work, "TMPDIR": work})
        except subprocess.TimeoutExpired:
            return False, f"TEST_TIMEOUT at test {i}"
        if p.returncode != 0:
            return False, f"CRASH at test {i}: {p.stderr[-150:]}"
        got = [x.rstrip() for x in p.stdout.strip().split("\n")]
        want = [x.rstrip() for x in t["output"].strip().split("\n")]
        if got != want:
            return False, f"WRONG at test {i}: {str(got)[:90]} != {str(want)[:90]}"
    return True, "ALL OK"

recs, t0 = [], time.time()
root = tempfile.mkdtemp(prefix="lcb-", dir="/mnt/ssd/memt-env/tmp")
for i, p in enumerate(problems):
    kind = p["tests"][0]["testtype"]
    prompt = (FUNC.format(q=p["question_content"], s=p["starter_code"])
              if kind == "functional" else STDIN.format(q=p["question_content"]))
    try:
        reply = ask(prompt)
    except Exception as e:
        reply = f"<<ERROR {type(e).__name__}: {e}>>"
    work = os.path.join(root, f"p{i}")
    os.makedirs(work, exist_ok=True)
    ok, why = judge(p, extract(reply), work)
    shutil.rmtree(work, ignore_errors=True)
    rec = {"question_id": p["question_id"], "platform": p["platform"],
           "difficulty": p["difficulty"], "testtype": kind,
           "passed": int(ok), "why": why, "reply_chars": len(reply)}
    recs.append(rec)
    with open(a.out_json + ".jsonl", "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if i % 10 == 0:
        acc = 100 * sum(x["passed"] for x in recs) / len(recs)
        print(f"  {i+1}/{len(problems)} pass@1={acc:.1f}% ({(time.time()-t0)/60:.1f}m) "
              f"{p['question_id']}/{p['difficulty']} {'PASS' if ok else why[:70]}", flush=True)
shutil.rmtree(root, ignore_errors=True)

def rate(sel):
    v = [x for x in recs if sel(x)]
    return {"n": len(v), "pass@1": round(100 * sum(x["passed"] for x in v) / len(v), 2)} if v else None

res = {"model": a.model_dir, "thinking": think, "max_new_tokens": a.max_new_tokens,
       "overall": rate(lambda x: True),
       "by_difficulty": {d: rate(lambda x, d=d: x["difficulty"] == d)
                         for d in ("easy", "medium", "hard")},
       "by_platform": {q: rate(lambda x, q=q: x["platform"] == q)
                       for q in ("atcoder", "leetcode")},
       "minutes": round((time.time() - t0) / 60, 1)}
json.dump(res, open(a.out_json, "w"), indent=2)
print(json.dumps(res, indent=2), flush=True)
print("OK", flush=True)
