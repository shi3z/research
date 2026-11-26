# SpikingBrain-7B の独立検証 — 「脳型スパイキング LLM」は何をしているのか

対象: [BICLab/SpikingBrain-7B](https://github.com/BICLab/SpikingBrain-7B) (MIT)、
重みは ModelScope の `Panyuqi/V1-7B-base`（15.24 GB、15 シャード）
技術報告: リポジトリ同梱 `SpikingBrain_Report_Eng.pdf`
測定日: 2026-11-26 ・ 環境: A100 80GB PCIe x1 (GPU 6)、torch 2.7.1 / transformers 4.55.2 /
flash-linear-attention 0.1 / flash_attn 2.7.4（リポジトリの pin に合わせた）

謳い文句は「4M トークンで 100倍以上の TTFT 高速化」「69% のスパイキング疎性」
「学習データ 2% 未満」。順に測った。

結論を先に書く。**論文は概ね正直で、README が盛っている。**

| 主張 | どこに書かれているか | 判定 |
| --- | --- | --- |
| Qwen2.5-7B からの変換モデル | 論文が明記 | **追認**（共有重みの cosine 0.97、無相関の対照は 0.0001） |
| 4M トークンで 100倍以上 | README・図の見出し | **外挿値**。論文本文は「曲線フィットからの保守的な見積り」と明記 |
| 1M トークンで 26.5倍 | 論文 Table 7 | 実測だが **H100 + 多デバイス列並列 (ZeCO+p2p)** 前提 |
| 単一デバイス 128k で 1.41–2.75倍 | 論文 Table 8 | これが普通の利用者の数字 |
| 69.15% の疎性 | 要約・図の見出し | **ビット単位の数字**。論文 A.8 が定義を書いている（下記） |
| スパイクで疎な加算が密な行列積を置き換える | Int2Spike README | **出荷コードでは起きない。** 該当経路はコメントアウトされ dense matmul が走る |
| 4M トークンを扱える | README | prefill はできる。だが **4096 トークンより前は取り出せない**（下記 needle） |

---

## 1. アーキテクチャ — 「スパイキング」はベースモデルには無い

`config.json` は **Qwen2.5-7B のものと寸法が一致する**。hidden 3584、28 層、
intermediate 18944、28/4 ヘッド、vocab 152064、rope_theta 1e6、
bos/eos 151643、norm_eps 1e-6、max_position 131072。
`hidden_act` だけ `"swish"` 対 `"silu"` だが**同じ関数**。

違うのは attention だけ:

| 層 | 実装 | 範囲 |
| --- | --- | --- |
| 奇数層 1,3,…,27（14層） | `FlashAttention` | **sliding window 4096** |
| 偶数層 0,2,…,26（14層） | `GatedLinearAttention` (fla の GLA) | 固定サイズの再帰状態 |

**グローバル attention が一層も無い。** `max_position_embeddings` は 131072 だが、
任意の過去トークンを厳密に参照する経路が存在しない。これは §4 で測る。

そして `hf_7B_model/activations.py` を読むと、**スパイキングのコードが一行も無い**。
sigmoid / swish / gelu / SwiGLU の Triton カーネル（flash-linear-attention からの
標準的なコピー）だけで、MLP は `down_proj(swish(gate_proj(x)) * up_proj(x))` —
Qwen2.5 の SwiGLU そのもの。

**ベースモデルとチャットモデルはスパイキングしていない。**
スパイキングは別配布の `W8ASpike` 変種にのみ存在する。

## 2. 「スパイク」の実体 — 出荷コードは dense matmul を回す

`W8ASpike/quant_linear.py` の `QuantLinear.forward`、実際に実行される行:

```python
spikes_int, vth = dynamic_spikes(x, self.k)
x = (spikes_int * vth).to(x.dtype)          # ← スパイクを閾値で掛け戻す
weight = self.weight_quantizer(self.weight)
out = torch.nn.functional.linear(x, weight, self.bias)   # ← 密な行列積
```

そのすぐ上に、疎な加算で行列積を置き換える本来の経路が**コメントアウトされて**置かれている
（`# NOTICE: can use spike_matmul func to substitute the matmul between spikes_int & weight.`）。

`dynamic_spikes` の中身は:

```python
vth = x.abs().mean([-1], keepdim=True).float() / k    # k = 3.0
spikes_int = (x / vth).round()
```

これは**トークンごとに動的な閾値を持つ活性量子化**である。
「スパイク」は 0/1 でも ±1 でもなく整数。しかも掛け戻して密に計算するので、
**疎性は実行時に何も買っていない**。

リポジトリの README 自身が「True-spiking は非同期・イベント駆動ハードウェアを要し、
本リポジトリの範囲外」と書いている。出荷コードはそれよりさらに一歩手前で、
疑似スパイクの行列積すら無効化されている。
論文の「エネルギー 97.7% 削減」は 45nm の公表電力値から
「スパイク誘起の重み加算」として**見積もった値**で、測定値ではない。

## 3. 由来 — Qwen2.5-7B の変換であることを重みで追認

論文はこれを隠していない。「We use an open-source Transformer checkpoint
(Qwen2.5-7B-base as an example)」「two models from the Qwen2.5-7B-base checkpoint」、
変換パイプライン、CPT 約 150B トークン。

重みで確認した（float64 で計算。float32 では 545M 要素の内積が丸めで壊れ、
`lm_head` の cosine が 1.10 という不可能な値になる）:

| テンソル | cosine | 相対誤差 | 最小二乗スケール α |
| --- | --: | --: | --: |
| `lm_head.weight` | 0.9162 | 0.4183 | 0.9496 |
| `layers.0.mlp.gate_proj.weight` | **0.9722** | 0.2343 | 0.9401 |
| `layers.27.mlp.down_proj.weight` | **0.9714** | 0.2379 | 0.9302 |
| **対照: 無相関なガウス行列 2 枚** | **−0.0001** | — | — |

グループ別の中央値（282 テンソル、fp32 計算なので cosine は目安）:

| テンソル群 | n | cosine 中央値 | 相対誤差 |
| --- | --: | --: | --: |
| `attn.q_proj.bias` | 28 | 0.999985 | 0.0805 |
| `attn.k_proj.bias` | 28 | 0.999886 | 0.0845 |
| `attn.v_proj.bias` | 28 | 0.999772 | 0.0797 |
| `model.norm.weight` | 1 | 0.999984 | 0.0699 |
| `mlp.up_proj.weight` | 28 | 0.981174 | 0.2470 |
| `mlp.down_proj.weight` | 28 | 0.980599 | 0.2472 |
| `mlp.gate_proj.weight` | 28 | 0.973677 | 0.2736 |
| `attn.o_proj.weight` | 28 | 0.977834 | 0.2126 |
| `attn.q_proj.weight` | 28 | 0.943557 | 0.3475 |
| `attn.k_proj.weight` | 28 | 0.939288 | 0.3778 |

**独立に学習された 2 つの 7B モデルの重み行列の cosine は 0 になる**（対照の −0.0001）。
0.97 は同一起源以外にありえない。バイアスはほぼ無傷（0.9998）、
attention 射影がいちばん動いている（GLA 変換で attention map を合わせるため。
論文もそう書いている）。α ≈ 0.93–0.95 の一様な縮みは weight decay の痕。

SpikingBrain 固有のテンソルは 113 個（`gk_proj`、`g_norm`、`attn_norm`、`mlp_norm`、
`model.embeddings`）。Qwen2.5 側にしか無いものが 57 個。

**これは論文の主張の追認であって反証ではない。**
ただし比較表で Qwen2.5-7B を「ベースライン」として並べるとき、
それが親チェックポイントであることは読者に伝わる形で書かれるべきだと思う。
「2% のデータ」も、ゼロから学習するコストではなく**変換コスト**である
（150B 対 Qwen2.5 の約 18T = 0.83%、論文の記述と整合）。

## 4. 本題 — 4M トークンを「流せる」が「取り出せない」

グローバル attention が無いモデルで、過去の情報は本当に残るのか。
needle-in-a-haystack を実行した。干し草は数字を一切含まない文章、
針は「金庫の暗証番号は NNNNN」、採点は数字の exact match（判定器不要）。
対照は親の Qwen2.5-7B（同じトークナイザ、同じ寸法、違うのは attention だけ）。
各セル 3 試行。

**SpikingBrain-7B の正解率 (%)**

| 文脈長 \ 針の位置 | 10% | 25% | 50% | 75% | 90% | 平均 |
| --: | --: | --: | --: | --: | --: | --: |
| 2,048 | 100 | 100 | 100 | 100 | 100 | **100.0** |
| 4,096 | 100 | 100 | 100 | 100 | 100 | **100.0** |
| 8,192 | 66.7 | 100 | 66.7 | 100 | 100 | **86.7** |
| 16,384 | **0** | **0** | **0** | 33.3 | 100 | **26.7** |
| 32,768 | **0** | **0** | **0** | **0** | 100 | **20.0** |

**16k 以上では、針が末尾 4096 トークン以内にあるときだけ当たる。**
末尾からの距離で並べ直すと境界がはっきり出る:

| 文脈長 | 針の位置 | 末尾からの距離 | 正解率 |
| --: | --: | --: | --: |
| 32,768 | 90% | 3,277 | **100%** |
| 16,384 | 90% | 1,638 | **100%** |
| 16,384 | 75% | 4,096 | 33.3%（ちょうど境界） |
| 16,384 | 50% | 8,192 | 0% |
| 32,768 | 75% | 8,192 | 0% |

これは `sliding_window: 4096` の値そのもの。
**16k 以上では GLA 層は取り出せる情報を実質的に運んでいない。**

8,192 では窓の外（距離 6,144 の 25% 位置など）でも当たるので、
GLA の再帰状態は確かに何かを保持している。だがその上に書き込まれる量が増えると失われる
— 固定サイズ状態の上書きそのもの。失敗の仕方も特徴的で、
針 `82420` に対して `8240` と答える（桁が落ちる）。
**無関係な答えではなく、劣化した答えを返す。**

**対照の Qwen2.5-7B は 25 セル全部で 100%。**

| 文脈長 \ 針の位置 | 10% | 25% | 50% | 75% | 90% | 平均 |
| --: | --: | --: | --: | --: | --: | --: |
| 2,048 – 32,768（全長） | 100 | 100 | 100 | 100 | 100 | **100.0** |

つまり課題そのものは易しく、**SpikingBrain の失敗はすべて attention の置き換えに帰属する。**
同じ重みから出発し、同じトークナイザで、違うのは
「sliding window 4096 + GLA」対「full attention」だけなのだから、
これ以上きれいな対照は作れない。

**この結果の意味。** 「4M トークンの prefill」は本当にできる。
だが**グローバルに参照しないからこそ安いのであって**、
4M トークン入れたところで 4096 トークンより前は取り出せない。
長文要約のように局所情報の集約で足りる課題と、
長文検索のように任意位置の厳密参照を要する課題を分けて考える必要がある。
論文は long-context の評価として LongBench 等を報告しているが、
needle 型の厳密検索の結果は見当たらない。

## 5. 再現

```sh
# 由来（重みのみ、GPU 不要）
python harness/provenance.py $SPIKINGBRAIN $QWEN25

# 長文検索
python harness/needle.py $MODEL out.json --lengths 2048,4096,8192,16384,32768

# prefill スケーリング
python harness/ttft.py $MODEL out.json --lengths 4096,...,131072

# 疎性（著者自身の dynamic_spikes を実活性に適用）
python harness/sparsity.py $MODEL wiki.test.raw out.json --tokens 4096

# MMLU-Pro（ベースモデル用・文字の対数確率）
python harness/mcq_base.py $MODEL mmlupro.parquet out.json --n 500
```

環境構築で引っかかる 2 点:

- **`flash-linear-attention==0.1` が transformers 4.55 と衝突する。**
  fla の `bitnet` 登録が transformers 自身の `bitnet` とぶつかり
  `ValueError: 'bitnet' is already used by a Transformers config` で死ぬ。
  リポジトリに `docker_build/patch_fla_bitnet.py` が同梱されていて、
  `exist_ok=True` を足すのが正解。
- **重みが `.bin`（pickle）で配布されている。** safetensors ではないので、
  読む側は `weights_only=True` を明示すること（本リポジトリのスクリプトはすべてそうしている）。
  `trust_remote_code=True` も必要なので、モデリングコードは読んでから動かすのが筋。

## ライセンス

リポジトリは MIT。重みは ModelScope 配布（`W8ASpike/README.md` は Apache-2.0 を記載）。
Qwen2.5-7B は Apache-2.0。評価データは再配布していない。
