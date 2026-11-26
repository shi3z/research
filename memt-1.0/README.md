# MemT-1.0 の独立検証 — 「ネイティブ記憶」は本当にコンテキストを置き換えるか

対象: [MemTensor/MemT-1.0-Flash](https://huggingface.co/MemTensor/MemT-1.0-Flash) および
[MemT-1.0-Nano](https://huggingface.co/MemTensor/MemT-1.0-Nano)
測定日: 2026-11-23 / 24 ・ 環境: A100 80GB PCIe x2 (GPU 4+6), transformers 5.12.1

MemT-1.0 は backbone の attention と並列に「Native Memory Block」を置き、
会話履歴を**固定サイズの記憶状態**に畳み込む。KV キャッシュのように履歴長に比例して
伸びないので、「66倍の記憶効率」「1.86倍の速度」を謳う。ここではその主張を
自分の手で測り直した。

結論を先に書く。

| 主張 | 判定 |
| --- | --- |
| 記憶状態が履歴長に対して定数サイズ | **正しい**（2K〜128K トークンで 41,944,024 バイト固定） |
| 66倍の記憶効率 | **Nano では正しい**（実測 64.0倍）。ただし 128K 時点の比較値 |
| 66倍は Flash にも当てはまる | **当てはまらない**（0.5〜8.6倍。公式起動設定の fp8 KV では記憶状態のほうが大きい） |
| full context より強い | **逆**。同一指標で full context が全カテゴリ勝利（F1 32.75 vs 55.47） |
| 一般能力はほぼ維持 | **維持どころか完全に同一**。記憶未投入時、Nano は backbone とビット単位で同じ出力 |

---

## 0. 一番効いた発見 — Nano は「記憶を入れない限り」backbone そのもの

コーディング・知識系ベンチマークを Nano と素の backbone (Qwen3.6-35B-A3B) で
走らせたところ、**全数値が完全一致**した。HumanEval 92.68% 対 92.68%、
MMLU-Pro は 14 カテゴリすべて一致、wikitext-2 の NLL は小数6桁まで一致
(1.788323)、LiveCodeBench v6 は難易度別・プラットフォーム別の全バケットまで一致。
生成テキストも一文字単位で同一だった（差分はサンドボックスの一時ディレクトリ名だけ）。

偶然ではありえないので重みを直接比較した:

```
nano tensors=753  backbone tensors=1045
shared-name tensors=693  memt-only=60
sampled 693 shared tensors: bit-identical=693  different=0
```

**共有する 693 テンソル全部がビット単位で同一。** MemT-1.0-Nano は
Qwen3.6-35B-A3B のテキスト塔を**凍結したまま**、記憶ブロックの
60 テンソルを足しただけのチェックポイントである。
(backbone 側にしか無い 352 テンソルは vision 塔 333 個と MTP ヘッド 19 個で、
Nano はテキスト専用かつ MTP を落としている。)

しかも ctx=4096 > `chunk_tokens`=2048 の素の forward でも NLL が一致する。
つまり記憶経路は `commit_memory` / `memory=` で明示的に文書を投入しない限り
**完全に不活性**で、読み出しが 0 を返すため恒等写像になる。

これの意味は二つある。

1. モデルカードの「一般能力をほぼ維持」は**自明に真**。維持しているのではなく、
   同じモデルだから同じ数字が出る。アーキテクチャ追加のコストはゼロではなく、
   測定不能（記憶を使わなければ存在しない）。
2. 逆に、論文が報告する劣化（Nano で LiveCodeBench v6 **−5.8**、MMLU-Pro −1.6 など）は
   **我々が入手できるこのチェックポイントからは再現しない**。記憶経路を動かした条件、
   あるいは別の学習段のモデルで測った数字と考えるほかない。
   論文の数字を引用するときは、この条件差を明示する必要がある。

## 1. 記憶効率 — 66倍は Nano 限定で、ほぼ正しい

記憶状態は `(1, num_memory_heads, head_dim, head_dim)` の **float32**、記憶層ごとに1枚。
設定ファイルから出る数字はモデルカードの自称値を正確に再現する（= 著者と同じ数え方）:

- Nano: 10 full-attention 層 x 16 head x 256² x 4B = 41,943,040 B = **10.49M パラメータ**（カード「10M」）
- Flash: 20 記憶層 x 64 head x 512² x 4B = **335.5M パラメータ**（カード「336M」）= 1.25 GiB

`save_state` の実測は 41,944,024 バイト（予測 +984 B = safetensors ヘッダ）。
**履歴長 2,048 / 8,192 / 32,768 / 65,536 / 131,072 のすべてで同一バイト数。**

| 履歴 | 記憶状態 | 同じ履歴を full ctx で持つ KV | 比 | commit | 記憶で質問 | full ctx で質問 | 速度比 |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 2,048 | 40.00 MiB | 40.0 MiB | 1.00x | 3.3s | 1.78s | 1.67s | 0.94x |
| 8,192 | 40.00 MiB | 160.0 MiB | 4.00x | 6.4s | 1.25s | 3.06s | **2.44x** |
| 32,768 | 40.00 MiB | 640.0 MiB | 16.0x | 27.6s | 1.29s | — | — |
| 65,536 | 40.00 MiB | 1,280.0 MiB | 32.0x | 51.6s | 1.46s | — | — |
| 131,072 | 40.00 MiB | 2,560.0 MiB | **64.0x** | 95.6s | 1.25s | — | — |

(32K 以降の full ctx 列は、同居ジョブがあり 80GB では OOM したため測っていない。
交差点は 8K で既に出ているので結論は変わらない。)

**隠れたコスト: commit は履歴長に比例する。** 128K トークンの取り込みに 95.6 秒。
「1.86倍速い」はクエリ時だけの比較で、この取り込み時間は含まれていない。
一度の取り込みで何度も質問するなら元は取れるが、一問一答なら 8K 以下では負ける。

## 2. 66倍は Flash には引き継がれない

Flash は 2 層 sliding(w=128) + 21 層 csa(1/4 圧縮) + 20 層 hca(1/128 圧縮) で、
KV 自体が既に圧縮されている。そこへ 1,280 MiB の記憶状態を足すので:

| Flash の比較対象 | KV サイズ | 記憶状態との比 |
| --- | --: | --: |
| bf16・圧縮 KV | 1.352 GiB | 1.1x |
| **fp8 KV（公式起動スクリプトの設定）** | 0.676 GiB | **0.5x（記憶状態のほうが大きい）** |
| 非圧縮 KV（疎読み出し） | 10.75 GiB | 8.6x |

論文は 66倍を Nano に限定して書いている。Flash のモデルカードの配置が
誤読を誘う形になっている、という指摘が正確なところ。

**Flash 自体はこの環境では動かせなかった**: 公式要件が 8 GPU の TP8/EP8/DP8、
単ノード・**非量子化 BF16 のみ**、`fp8_e4m3` KV キャッシュ、SGLang 0.5.16 + パッチ3本。
536.4 GiB / 8 = 67.1 GiB/GPU に対し使用可能 79 GiB で机上は入るが、本番モデルが
載っている GPU を含めて全台専有する必要があり、かつ A100 は sm_80 で FP8 がネイティブに無い。

## 3. 本題 — full context に負ける

LoCoMo、会話 2 本、**304 問**、記憶アームと full-context アームで
**完全に同一の指標**（SQuAD 式正規化のトークン F1 / EM）。違うのは
「記憶がどこに置かれているか」だけ。

| | 記憶アーム | full-context アーム | 差 |
| --- | --: | --: | --: |
| 全体 token-F1 | 32.75 | **55.47** | +22.7 |
| 敵対的カテゴリを除く (n=233) | 34.58 | **46.19** | +11.6 |
| 全体 EM | 15.79 | **35.86** | +20.1 |
| 平均クエリ遅延 | **1.18s** | 3.30s | 記憶が 2.8倍速い |

カテゴリ別 F1（左=記憶 / 右=full context）:

| カテゴリ | 記憶 | full ctx |
| --- | --: | --: |
| 1 マルチホップ | 40.81 | 51.59 |
| 2 時間推論 | 18.88 | 33.49 |
| 3 オープンドメイン | 16.66 | 17.95 |
| 4 シングルホップ | 42.95 | 54.39 |
| 5 敵対的（答えが無い） | 26.76 | **85.92** |

**全カテゴリで full context が勝つ。** 敗因は二つに切り分けられた。

- **「知らない」と言えず作話する。** カテゴリ5（会話に答えが存在しない問題）の
  EM は 25.35 対 84.51。full context は「No information available.」と答えられるが、
  記憶アームは何かを答えてしまう。固定サイズ状態には「無い」を表現する術がない。
- **絶対日付が消える。** 「2023年5月7日」に対して「Last Friday」、
  「2023年6月」に対して「Next month」と答える。セッションのタイムスタンプが
  commit の時点で失われている。

注意: 論文は gpt-4.1-mini 判定 + OmniMemEval プロンプトなので、
**絶対値は論文の LoCoMo 57.53 と比較できない**。ここで意味があるのは
同一指標・同一モデルでの**アーム間の差**だけである。

論文自身の構造も同じ方向を向いている。「パラメトリック記憶手法の中で2位」という
カテゴリは最高の対抗手法が 38.08 点の世界であり、同じ backbone の
テキスト記憶手法と素の full context は MemT を 14〜18 点上回っている。

## 4. プログラミング能力

論文の一般能力表（対 backbone の差分）:

| | Nano | Flash |
| --- | --: | --: |
| MMLU-Pro | −1.6 | −1.8 |
| GPQA | −1.5 | −0.6 |
| IFEval | +3.3 | +0.9 |
| LongBench v2 | −2.0 | +0.2 |
| BigCodeBench-Hard | **+2.0** | **+2.0** |
| LiveCodeBench v6 | **−5.8** | −0.7 |
| 平均 | −0.9 | **±0.0** |

Nano の最大の劣化はコード生成で、表全体でも最大。一方 BigCodeBench-Hard は +2.0。
「ゼロからアルゴリズムを書く」が落ち「複雑な実務コードを組む」は上がる、という分かれ方。
Flash（active 17B）では消えている。

ただし **§0 の通り、この −5.8 は我々の手元のチェックポイントでは再現しない**。
同一条件（thinking off、1536 トークン）で測った我々の数字は両者完全一致である:

| | MemT-1.0-Nano | 素の backbone | 差 |
| --- | --: | --: | --: |
| HumanEval pass@1 (n=164) | 92.68 | 92.68 | 0.00 |
| LiveCodeBench v6 pass@1 (n=175) | 29.14 | 29.14 | 0.00 |
| └ easy (43) / medium (52) / hard (80) | 83.72 / 25.0 / 2.5 | 83.72 / 25.0 / 2.5 | 0.00 |
| MMLU-Pro 文字対数確率 (n=490) | 59.59 | 59.59 | 0.00 |
| wikitext-2 PPL (ctx 4096) | 5.9794 | 5.9794 | 0.00 |

LiveCodeBench は **thinking off・1536 トークン**で走らせているので、
絶対値は論文の 88.06 と比較できない（論文は確実に thinking on）。
意味があるのは差分で、その差分は 0.00 である。
HumanEval は 92.68% で飽和しており判別力が無い。

**論文側のいちばん重要な表 (Table 13)。** agentic coding では MemT は「経験なし」より
確かに強い（SWE-bench Verified 66.07 → 69.33、pass^3 54.00 → 61.60）。
しかし同じ経験を**テキストでコンテキストに入れたほうが強い**:

| SWE-bench Multilingual (300タスク) | Avg Success | pass^3 |
| --- | --: | --: |
| 経験なし | 60.00 | 45.33 |
| **Textual Experience: Fixed Rank-1** | **64.22** | **56.33** |
| MemT-1.0-Nano (native memory) | 62.89 | 53.67 |

Fixed Rank-1 は oracle 選択ではない公平なベースラインで、それでも native memory に
+1.3 / +2.7 で勝つ。(Per-Task Best 70.00 は oracle 選択であり、論文もそう書いている。)
記憶ベンチと同じ構図がプログラミングでも成立しているが、**差は 16 点から 1.3 点まで縮む**。
手続き的経験（「こういうバグはこう直す」）は具体的事実より圧縮に耐える、と読める。

## 5. 測り方と再現

```sh
# 設定だけで出る記憶効率（重み不要）
python harness/state_vs_kv.py

# 記憶状態の実測サイズと効率
python harness/bench_state.py $NANO --mem 36,33

# 本題: 同一指標で記憶アーム対 full-context アーム
python harness/locomo.py $NANO locomo10.json out.json --arm memory
python harness/locomo.py $NANO locomo10.json out.json --arm fullctx

# 一般能力（判定器なし）
python harness/ppl.py  $MODEL wiki.test.raw out.json --ctx 4096 --chunks 24
python harness/mcq.py  $MODEL mmlupro.parquet out.json --n 500
python harness/humaneval.py $MODEL humaneval.parquet out.json
python harness/lcb.py  $MODEL lcb_test6.jsonl out.json
```

全ハーネスは **ディスクオフロードが起きたら中断する**チェックを持つ。
`max_memory` が足りないと一部パラメータが `meta` デバイスに載り、
静かに別物を測ることになるため:

```python
meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
if meta:
    raise SystemExit(f"ABORT: {len(meta)} params on meta; raise --mem")
```

踏んだ落とし穴は [NOTES.md](NOTES.md) に置いた。

## データとライセンス

LoCoMo は CC BY-NC 4.0。評価にのみ使用し、再配布していない。
MMLU-Pro / HumanEval / LiveCodeBench v6 も各自の配布元から取得すること。
結果の生 JSON は [results/](results/) にある。
