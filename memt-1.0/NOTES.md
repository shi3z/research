# 測定で踏んだ落とし穴

MemT-1.0 を測るときに時間を食った箇所。同じことをする人向け。

## 1. `max_memory` が足りないと静かに別物を測る

`max_memory={i: "34GiB"}` を 2 枚（計 68 GiB）に対して重みは 65.4 GiB。
余裕があるはずが、**23 パラメータが `meta` デバイスに載った**（ディスクオフロード）。
例外は出ず、生成も通り、数字も出る。別のモデルを測っていることになる。

全ハーネスの起動時にこれを入れた:

```python
meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
if meta:
    raise SystemExit(f"ABORT: {len(meta)} params on meta; raise --mem")
```

落ち着いた設定は `--mem 36,33`（69 GiB）+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 2. モデルカード推奨の `attn_implementation="eager"` は無駄にメモリを食う

`sdpa` ならオフロードゼロで載る。eager だとオフロードが発生して §1 を踏む。

## 3. `<think>` が出る

`enable_thinking=False` を明示しないと思考トークンを出し、16 トークンの予算では
答えに到達せず切れる。全呼び出しで明示すること。

## 4. transformers 5.12.1 の `apply_chat_template`

`return_tensors="pt"` は **BatchEncoding を返す**（テンソルではない）。
`.shape` を触ると `AttributeError` になり、我々の最初の MMLU-Pro 実行は
backbone 側が 280 問すべて `<<ERROR AttributeError: >>` で 0.0 分で「完走」した。

```python
enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                              enable_thinking=False, tokenize=True,
                              return_dict=True, return_tensors="pt")
ids = enc["input_ids"].to(dev)
```

## 5. 選択式を generate + 正規表現で解くと測れない

「文字だけで答えよ」を両モデルが無視し、Nano 側は 280 問中 252 問が解析不能
（散文を書いて 24 トークンで切れる）。**文字の対数確率の argmax に替えた**:
forward 1 回、生成なし、正規表現なし。`"A"` と `" A"` の両方の単一トークン形を採点する。

## 6. perplexity が OOM する

語彙 248,320 なので ctx 4096 のロジットは bf16 で 2 GiB、fp32 にすると 4 GiB、
`cross_entropy` がもう 1 枚要求して 3.79 GiB の確保に失敗する。
**256 行ずつスライスして損失を足す**:

```python
nll = 0.0
for s in range(0, logits.shape[0], 256):
    part = logits[s: s + 256].float()
    nll += F.cross_entropy(part, targets[s: s + 256], reduction="sum").item()
    del part
```

## 7. `pgrep -f` の待ちループが自分自身に一致して無限ループする

`pgrep -f 'aria2c ... foo.aria2.txt'` を待ち条件にすると、**その pgrep を起動した
シェル自身のコマンドラインにもその文字列が含まれる**ので永遠に真になる。
ジョブは完了しているのにループが終わらない。

待つなら数値 PID で `kill -0`:

```sh
while kill -0 "$1" 2>/dev/null; do sleep 60; done
```

プロセス確認は `ps … | grep -F` で目視してから単一の数値 PID を kill する。

## 8. `memory=` は最初の chat だけ

`chat(query, memory=[...])` の `memory=` が受け付けられるのは
**セッションの最初の chat のみ**。2 回目以降は黙って無視される。
文書を追加したいなら `commit_memory` を使う。

## 9. HumanEval は飽和していて使えない

Nano で 164 問 92.68%。コード生成が壊れていないことの確認にはなるが、
論文が主張する −5.8 のような差を見るには判別力が無い。
LiveCodeBench v6（hard 80 問を含む 175 問）に替えた。

## 10. LiveCodeBench のテストケースは二重符号化

`json → zlib/base64 → pickle` で、しかも pickle が返すのが JSON **文字列**なので
もう一度 `json.loads` が必要:

```python
def decode(s):
    try: v = json.loads(s)
    except Exception:
        raw = zlib.decompress(base64.b64decode(s.encode()))
        try: v = json.loads(raw)
        except Exception: v = pickle.loads(raw)
    return json.loads(v) if isinstance(v, str) else v
```
