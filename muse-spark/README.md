# 816B MoE を 1枚の A100 で回す — Muse-Spark-Open-816B-A42B

対象: `meta-models/Muse-Spark-Open-816B-A42B`（Q4_K_S GGUF、22 シャード、438 GiB）
環境: A100 80GB x1 + ホスト RAM、llama.cpp master ワークツリー
測定: 2026-10-16

dense 重みを GPU に、256 エキスパートの MoE をホスト RAM に置く Strata 式で
816B (active 42B) を 1 枚の A100 に載せ、base / DFlash2 ドラフト / 自作 DSpark ドラフトを比較した。

## 結果

| 構成 | decode tok/s | 備考 |
| --- | ---: | --- |
| base | 5.7–6.5 | prefill 10–40 tok/s |
| DFlash2 ドラフト | 5.6 | 受理率 61%、5.25 tok/cycle。コードでは 3.2 |
| 自作 DSpark ドラフト | 4.2–5.0 | 受理率 37–51%（モデルカードの 28–42% と整合） |
| 32 スレッド + 6層分のエキスパートを GPU に (`--n-cpu-moe 56`) | **7.3–7.6** | |
| **静的ホットエキスパート分割** | **8.3–9.2** | 最良 |

**投機デコードはここでは効かない。** 受理率は仕様どおり出ているのに
base より速くならない。律速がホスト RAM のエキスパート読み出しであって
GPU の forward 回数ではないため、ドラフト 1 本あたりの検証コストが下がらない。

**llama.cpp の GPU LRU エキスパートキャッシュ (`--moe-cache-mib`) はここでは遅い**
（3.8–4.5 tok/s）。ミスが PCIe アップロード（約 1.2 ms/エキスパート）になり、
CPU で計算する約 0.35 ms に負ける。しかも plain host buffer を要求する。

**ルーティングは確かに偏っている**: 層あたり上位 27/256 エキスパートが選択の 69%。
これを使った静的ホットエキスパート分割（デコードバッチ 4 以下限定）が最良の 8.3–9.2 tok/s。

## 必須設定（どれか欠けると性能が出ない）

```
--load-mode none
numactl --interleave=all
--numa distribute
sysctl kernel.numa_balancing=0      # 終わったら 1 に戻す
THREADS=24                           # 44 ではない
```

## 時間を食った教訓 2 つ

**ホットリストは decode のルーティングから取らなければならない。**
prefill コーパスから作ったリストでは効かない。`GGML_MMID_HIST` を素の実行に付けて
デコード時のヒストグラムを集める。

**tmux は環境変数を継承しない。** 起動スクリプトから GPU/THREADS を明示的に渡すこと。

GPU が他のジョブと共有されていると 3 倍遅くなる（専有の確認を先にする）。
