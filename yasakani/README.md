# YASAKANI `.yskn` 形式の独立検証と A100 (sm_80) への移植

対象: [`yasakani-dev/yasakani`](https://github.com/yasakani-dev/yasakani) (Apache-2.0)、
公開重み `yasakani-dev/Qwen3.8-27B-YASAKANI-{quality,speed}`
測定日: 2026-11-14 ・ 環境: A100 80GB PCIe (GPU 7), CUDA 12.8

## 1. 形式は再現でき、公開重みは本物だった

**DLL の逆アセンブルは不要。** リポジトリは `src/*.cu` 24 ファイルの完全な CUDA/C++ 実装、
`docs/FORMAT.md` のバイトレベル仕様、変換器 `tools/convert/yskn_convert.py`、
純 numpy のリーダ `tools/yskn_decode.py` を同梱している。
release zip の Windows `yskn.dll` はビルド済みバイナリに過ぎない。

検証: **仕様書だけから独立に numpy 実装を書き**、`layers.7.self_attn.k_proj` を復号して
元の Qwen3.8-27B テンソル（safetensors の 10 MB バイト範囲で取得）と比較した。

| 指標 | 実測 | 仕様の記載 (4 bpw) |
| --- | ---: | --- |
| 相対 Frobenius 誤差 | **0.0910** | 0.085–0.106 |
| 要素ごとの相関 | **0.9959** | — |

仕様の範囲内。**形式は再現可能で、公開されている重みは本物。**

実体は QTIP (arXiv:2406.11235) の trellis 符号化 + QuIP#/QuaRot のブロック512 Hadamard 回転
（符号は mulberry32 seed 7 の擬似乱数で非保存）+ GPTQ/LDLQ 誤差フィードバック +
YAQA の出力側（勾配）統計、テンソル単位で 3 または 4 bpw 割り当て。
12 ビットスライド状態、行列ごとに 4096x2 のガウス値表 1 枚、行ごとに fp16 スケール 1 個、
2 重みあたり 6 または 8 ビット。quality モデルは 26.87 B パラメータを 3.659 bpw (12.37 GB)。
後段の層が 4 bpw、前段・中段 MLP と linear-attn の in_proj が 3 bpw。

KV5（同じ trellis を KV キャッシュに適用）は 22 KiB/token 対 FP16 の 64 KiB で
KL 増加は最大 +0.00052、8 ビットより遅くもない。

## 2. A100 への移植 — 測定値

branch `a100-sm80`。ロード 6 秒、262144 コンテキストで VRAM 22 GB、
4 タスク平均 **202.6 tok/s**（explain 185.8 / code 276.6 / story 102.8 / proof 245.3、
A6000 fork と同じ tokens/pass なので直接比較可能）、look-ahead オフで 54 tok/s・1 pass 18.6 ms。
**look-ahead はビット単位で同一のテキストを返す** — 整数厳密 GEMM の性質が sm_80 でも成立している。

### 上流に返したバグ: PDL がビルド側で有効のままだった

A6000 fork は programmatic dependent launch のデバイス側（`pdl.cuh` の griddepcontrol、
sm_90 未満では何もコンパイルされない）を guard していたが、`pdl_enabled()` が
true を返したままだった。そのため `launch_k` が全カーネルを
programmatic-serialization 属性付きの `cudaLaunchKernelEx` に通し、
pre-sm_90 のカードはそれを honour できない。

`pdl_enabled()` にデバイスの compute capability major を問わせると:

| | 修正前 | 修正後 | 差 |
| --- | ---: | ---: | ---: |
| 1 pass | 18.63 ms | 17.51 ms | **−6.0%** |
| 4タスク bench | 202.6 tok/s | 210.9 tok/s | **+4.1%** |

selfcheck と tokens-per-pass は不変。`shi3z/yasakani-not5090` に `cac9ad4` として push。

## 3. 残り時間はどこに行っているか

1 pass = 11.81 GB の符号読み出しで 17.5 ms。この A100 の
ストリーミング読み出し下限は **6.95 ms**（12 GiB 読み出しカーネルで 1698 GB/s、
SM あたり 4 CTA 以上。1 CTA で 830、2 で 1362）。

`lnord_kernel`（重み GEMV）が GPU 時間の 85%、実効 875 GB/s。
内側ループを 60 行で忠実に再構成すると 1229 GB/s が上限で、
**符号のロード単体は 1747 GB/s 出る**。つまり差の 40% は
**形式が要求する「符号 1 バイトあたり 1 回のデータ依存 16 ビット shared gather」**のコスト。
per-CTA の lut_copy、alpha プロローグ、エピローグを足すと 974〜1038 GB/s まで落ちるので、
カーネル内の余地は約 1.19倍 = bench で 10% 程度。CTA 寿命はほぼ無関係
（warp あたり 4 block iteration で 974、64 で 1038）。

クロックは 300 W 中 277 W で最大、スロットリングなし。**時間は実際の仕事**。

**測って外れた 4 つの改善案**（再導出しないために記録）:

| 案 | 結果 |
| --- | --- |
| A100 の広い shared memory に LUT を複製 | **悪化。** 全コピーがバンク0起点なので競合は不変、occupancy が落ちる (1369 → 810 → 452 GB/s for 1/2/4 copies) |
| 32 ビット表エントリ | **悪化** (1248 → 922) |
| より深い prefetch | ノイズ |
| 他の block 形状 | このカーネルの形状での純読み出しが既に 1617 |

**5090 が 1 pass で 2.4倍速いのは帯域ではなく SM数×クロックが 2.7倍だから。**
A100 のほうが帯域は広い。

## 4. この形式のトレードオフは Ampere でだけ姿を見せる

11.81 GB の trellis 符号を gather 付きで読むと A100 で 13.5 ms。
同じモデルの素の 4 ビット・スケールのみ形式（約 13.5 GB）なら
復号作業がほぼ無いまま 1747 GB/s で読めて **7.7 ms = 1.75倍速い**（品質は下がる）。

つまり **trellis 形式はメモリを買って速度を売っている**が、
5090 では復号が読み出しの陰に隠れてこのトレードが見えない。
Ampere 世代で動かすなら、エキスパートを FP4 + スケール（表なし）に保つほうが正しい。

## 落とし穴（このボックス固有ではないもの）

**`--device N` では足りない。** エンジンは `--device` が何であれ
device 0 でカーネルの dynamic shared memory opt-in を行うので、
GPU 0 が埋まっているボックスでは
`launch_q35_atx_pk: cudaFuncSetAttribute failed: out of memory` で死ぬ。
`CUDA_VISIBLE_DEVICES` を使うこと。

**Python パッケージが環境変数 `YSKN_*` を消す。** `RuntimeProfile.apply()` は
シェルが設定した `YSKN_*` のうち自前の 17 エントリのプロファイルに無いものを
全部削除してからプロファイルを適用する。`env YSKN_...=x` での調整は
**何も測っていない**（47 構成のスイープを 1 回丸ごと無駄にした）。
効くのは `Engine(pack, profile=RuntimeProfile.default(pack, overrides={...}))` だけ。
`yasakani bench`/`chat` は "note: ignored N YSKN_* setting(s)" と出すが Engine API は黙る。
ノブが効いていることは**必ず失敗するはずの対照**で確認する
（例: `YSKN_Q35_KV_FP16=1` → "Rotated INT8 runtime requires INT8 storage"）。
