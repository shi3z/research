# NariQ v1: what the format is, and what the reference encoder does

Sources: `nari-labs/nari-llama.cpp` at 8e0e016 (clone: /mnt/ssd/git/nari-llama.cpp) and
`turboderp-org/exllamav3` at 8c593e4 (clone: /mnt/ssd/git/exllamav3).  The reference artefacts are the
published `nari-labs/Qwen3.8-27B-NariQ-GGUF` files; byte ranges of the 3.0bpw file are cached under
/mnt/ssd/nariq-ref (header, `blk.0.ffn_gate`, `blk.0.ssm_alpha`).

## 1. The format did not need reverse engineering from the kernels

`nariq/FORMAT.md` in the fork documents version 1, and `ggml/src/ggml-trellis.cpp` is a plain-C++ host
decoder for it — the same one `nariq/decode_reference.py` calls through ctypes.  So the encoder is checked
against the decoder that will actually read its output (`nariq/oracle.py`), not against a second guess.

The bitstream, tile ring and packing are **EXL3's**, bit for bit: a 16x16 tile is a ring of 256 positions,
position p's 16-bit state is ring bits [(p+1)k - 16, (p+1)k) read MSB first and wrapped (tail-biting), the
ring is stored as `8k` uint32 words with ring bit 0 at the MSB of word 0, and the ring-position -> element
map is exllamav3's tensor-core fragment order.  NariQ's element map is EXL3's **transposed**: a NariQ tile
element is (output row, input column), an EXL3 one is (input row, output column).  Codes are stored
output-tile-major: tile (i, J) at word offset `(J * K/16 + i) * 8k`.

## 2. The codebook is NOT EXL3's mul1

Both multiply the state by `0x83DCD12D`.  EXL3's mul1 then sums the four whole bytes of the product and maps
`1024 + bytesum` through fp16 (`k_inv = 0x1eee`, `k_bias = 0xc931`) to a unit-variance value.  NariQ instead
sums the four bytes' **top 6 bits** and subtracts 126:

```c
uint32_t x = state * 0x83DCD12Du;
int v = ((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63) - 126;   // [-126, 126]
```

so the grid is integral (its INT8 CUDA path multiplies it directly) with std
`sqrt(4 * (2^12 - 1) / 12) = 36.9459`.  Correlation with mul1 is 0.998 — the same sum, coarsened — but the
maps differ by up to 0.1 unit-variance units, so a Viterbi scored with mul1 would not be scored with NariQ's
codebook.  `nariq.format.codebook()` is verified against `ggml_trellis_decode_rows` for all of TRL1..TRL6.

The fork's own `nariq/NOTICE` confirms the lineage after the fact: it credits exllamav3 for "its trellis
code packing and in-tile order, its tensor-core order within a tile, and its MUL1 multiplier for the
INT8 grid" -- the multiplier, not the codebook, which is exactly where the two differ.

## 3. Side transforms and scales

`W = P_out diag(post) T_rows^T W_hat T_cols diag(pre) P_in`, `W_hat = grid * s16`, with `s16` one f16 scale
per 16 **input** columns (shared by every output row), `pre` one f32 per input column, `post = sv * row_scale`
one f32 per output row.  Descriptor kinds: 0 identity, 1 normalised block Sylvester Hadamard, 2 whole-slice
Hadamard (Paley factors 12/20/68 for the non-power-of-two part).

The published 3.0bpw file uses **only kinds 0 and 1**, one part, one shard — no whole-slice Hadamard, no fused
parts.  Blocks: the hidden dimension 5120 always gets 128, while 10240 and 6144 get 2048, 17408 gets 1024,
the lm_head bands' 15488 gets 128 and 15616 gets 256.  `pin`/`pout` appear only where llama.cpp's linear
attention wants a different V-head order (64 `trellis_pin`, 240 `trellis_pout`); the converter builds those
itself, so an encoder that writes HF order never needs them.

Per-module bitrates in the 3.0bpw release are mixed TRL2..TRL5 (plus TRL6 for the first lm_head band):
`ssm_alpha`/`ssm_beta` are always 5-bit, `attn_v`/`attn_k` mostly 5-bit, the FFN mostly 2-3 bit.  The lm_head
is 16 frequency bands, 6/4/3/2,2,2... bits, with `output.vocab_gather`; `token_embd` is a single TRL2 matrix.

## 4. What the reference encoder's scales actually look like

Measured on `blk.0.ffn_gate` (TRL2, K=5120, N=17408, desc `[1,1024,1,1,128,1,1,17408]`), decoded bit-exactly
and compared with the original BF16 `model.language_model.layers.0.mlp.gate_proj.weight`:

| quantity | value |
| --- | --- |
| reference reconstruction error, Frobenius | **0.3673** relative |
| grid vs. its own rotated target | 0.3552 relative, correlation 0.938 |
| `pre` | random signs, \|pre\| in [0.979, 1.025], uncorrelated with column norms (r = 0.007) |
| `post` | random signs, \|post\| in [0.976, 1.026], uncorrelated with row norms (r = 0.006) |
| `s16` | 2.79e-4 .. 5.51e-4, rising monotonically with the input block index |
| rotated per-block rms | flat: 100.1e-4 .. 102.4e-4 (±1%) |

So the magnitude lives entirely in `s16`; `pre` and `post` are sign vectors with a ±2.5% modulation that
tracks neither row nor column norms.  Because `s16` rises ~45% while the data's rms is flat, the target grid's
std falls from 35.8 (block 0) to 24.8 (block 319) and the **per-block relative error degrades monotonically
from 0.278 to 0.580**.  A scale that held every block at the codebook's std would put every block near 0.278,
i.e. about 1.28x lower Frobenius error overall.

That is *not* evidence the reference is wrong: a coarser scale where a channel matters less is what an
importance- (Hessian-) weighted allocation looks like, and after the 128-wide column Hadamard such a weight is
constant inside each 128-block, which is what `s16` shows (within-group cv 1%).  Frobenius error cannot tell
the two apart, which is exactly why bitstream compatibility must never be reported as paper-equivalent
quality.

## 5. Consequences for the encoder

* The EXL3 tail-biting Viterbi can be reused unchanged in structure; only the codebook scoring changes.  Its
  state convention: cost arrays are indexed by the low `16-k` bits of the window, candidate k new bits go in
  the top, and the predecessor is `state >> k`.
* `quantize_tiles_use_optimized` is false for sm_80, so an A100 runs the generic kernel — the one that costs
  `2 * (65536 >> k)` halves of shared memory and `L * (65536 >> k)` uint16 of global history per tile.
* Nothing in the encoder needs the model architecture: a coded module is (W, bits, row/col transform blocks).

## 6. What the reference's sign vectors and scale modulation are (measured on the published 3.0bpw file)

**One sign stream per side, shared by every module.**  `sign(trellis_pre)` is identical for every coded
module with the same K -- every layer, every kind -- and the 5120-entry vector is exactly the first 5120
entries of the 17408-entry one.  `sign(trellis_post)` is a second stream of the same kind (agreement with
the input stream is 0.49, i.e. independent).  A decoder cannot tell: each tensor carries its own `pre`.
What it buys is at inference time -- `H diag(su) x` is then the same vector for every module that
consumes that activation, so a runtime can rotate once per layer instead of once per module, which is
probably also why the hidden dimension is rotated in blocks of 128 rather than the 1024 its size allows.
`nariq.encode.signs` reproduces the structure (prefix-sharing, two streams) with its own seed.

**The +-2.5% magnitude modulation of `pre` and `post` is not a scale refit.**  The format's "input signs
and scales" makes a per-input-column rescale free, and it is separable: `W = diag(post) T_r^T W_hat T_c
diag(pre)` means column j of W is `pre[j]` times column j of the rest, so the least-squares optimum is
`pre[j] = <W[:,j], M[:,j]> / <M[:,j], M[:,j]>`.  On `blk.0.ffn_gate` that optimum has mean 0.9215 and std
0.0552, while the file's own `|pre|` has mean 1.0024 and std 0.0069, and the two are uncorrelated
(r = 0.04 on the magnitudes, though 0.998 on the signed values -- the signs dominate).  So the reference
leaves that refit unused: its reconstruction is systematically ~8% long in the column-projection sense,
consistent with the 0.953 global shrinkage measured in section 4.

What the modulation *is* remains unexplained.  It is not the original or rotated column norms (r = 0.007),
not shared between modules that see the same activations (r = 0.02), not shared across layers (r = 0.015),
and it is stored at f16 granularity (multiples of 1/2048 near 1).  Its std, 0.0069, is close to
1/sqrt(2N) = 0.0054 for N = 17408, which is what the sampling noise of a per-column norm after a Hadamard
would look like, but no per-column norm we computed matches it.

## 7. Baselines measured here

| what | value |
| --- | --- |
| published 3.0bpw, wikitext-2 PPL (ctx 2048, 32 chunks, `llama-perplexity`) | **6.3523 +- 0.0835** |
| published 3.0bpw file sha256 | `1eaad934...be8f4419`, matching `nariq/run.sh`'s pin |
| its bitrates | TRL2..TRL5 per module, TRL6 for the first lm_head band; 16 banded lm_head bands |
| `blk.0.ffn_gate` (TRL2) reconstruction error | 0.3673 relative Frobenius |
| ours at the same shape, bits and transforms | 0.2617 |

The fork's own `nariq/cuda-qualification.md` makes the same point this file's section 4 does, about its
own KL numbers: "References come from `nariq/decode_reference.py`; this is a same-weight runtime check on
a short sample, not a quality measurement."

## 8. The first whole-model artefact, measured

Qwen3.8-27B (26.889 G parameters: 64 blocks, 48 of them linear-attention, plus a 248,320-token
embedding and an untied head).  One A100, 48.9 minutes for 497 coded modules; every module's packed
bytes were sampled and decoded back to the grid the Viterbi chose.

| model | bytes | bits/weight | wikitext-2 PPL (ctx 2048, 32 chunks) | mean KL vs BF16 | top-1 |
| --- | ---: | ---: | --- | ---: | ---: |
| BF16 | 53,808,276,736 | 16 | **6.1963 +- 0.0831** | - | - |
| published NariQ 3.0bpw | 10,123,626,592 | 3.012 | **6.3523 +- 0.0835** | 0.1089 | 86.2% |
| ours, uniform 3-bit + 2-bit embedding + BF16 head | 12,048,789,056 | 3.585 | **6.3686 +- 0.0861** | 0.0681 | 90.8% |

Read that carefully.  Our first file is *larger* (3.59 against 3.01 bits per weight, because its head is
still BF16) and lands a hair behind on perplexity, while being closer on the KL sample.  The KL is 65
rows from one prompt and the perplexity is 65,536 tokens, so the perplexity is the number to trust:
**a uniform bitrate at 3.59 bpw does not beat the reference's mixed 2-6 bit allocation at 3.01 bpw.**
That is the gap per-module allocation and a coded head exist to close, and the reason this file is a
checkpoint in the work rather than a result about NariQ quality.

The KL protocol is the fork's own `test-trellis-model` + `compare_logits.py`, with the candidate
teacher-forced on the reference's tokens (two different models diverge within a few tokens otherwise,
and `compare_logits` requires the same sequence).  The BF16 reference does not fit one A100, so its
capture runs on the CPU; the comparison is of logits, not of devices.

## 9. Per-module bit allocation

The loss a module causes is `e_m^2 S_m` with `e_m` its activation-weighted relative error and
`S_m = sum_j h_j ||W_m[:,j]||^2` its weighted energy, and a trellis halves `e` per bit, so minimising
`sum_m e_m(b_m)^2 S_m` under a byte budget is a water-filling: give the next bit to whichever module
buys the most loss per byte.  `nariq/tools/bit_plan.py` does that from one uniform run's errors plus the
checkpoint and the imatrix.

At a 3.0 bpw target it chooses, without being told anything about the architecture: 4-6 bits for the
48x5120 DeltaNet gates (tiny matrices, so protection is nearly free), a median of 4 for `v_proj`, 3 for
`gate_proj`, and 2 for `down_proj` and `out_proj`.  The published file's own allocation is 5 bits for
`ssm_alpha`/`ssm_beta`, mostly 5 for `attn_v`/`attn_k`, and 2-3 across the FFN -- the same shape,
arrived at independently.

## 9b. What the encoder achieves, per bitrate, on real weights

Median relative Frobenius error over the modules a 3.0 bpw allocation gave each bitrate, with no
calibration weighting (so this is the encoder against the plain objective), on Qwen3.8-27B:

| bits | modules | median relative error | 2^-bits |
| ---: | ---: | ---: | ---: |
| 2 | 50 | 0.2620 | 0.2500 |
| 3 | 14 | 0.1320 | 0.1250 |
| 4 | 13 | 0.0670 | 0.0625 |
| 5 | 20 | 0.0345 | 0.0312 |
| 6 | 1 | 0.0181 | 0.0156 |

`2^-bits` is the Gaussian rate-distortion bound for an infinite-dimensional quantizer, so a trellis of
16 states' memory landing 5-8% above it is the expected gap, and it holds on real weight matrices and
not just on Gaussian targets.  For comparison, the best 4-level *scalar* quantizer (Lloyd-Max) on the
same data is 0.3428, which is what the trellis structure buys at 2 bits.

## 10. The calibration weighting is a measured loss, not a gain

Three whole-model runs plus one graft, all Qwen3.8-27B, all wikitext-2 (ctx 2048, 32 chunks).  The
graft (`nariq/tools/graft.py`) is run A's blocks with run B's coded head, so each row differs from its
neighbour in exactly one thing:

| configuration | bits/weight | PPL | what it isolates |
| --- | ---: | --- | --- |
| A: uniform 3-bit blocks, BF16 head | 3.585 | 6.3686 | - |
| graft: A's blocks + a 3-bit coded head | 2.970 | 6.4448 | coding the head: **+0.076 PPL for -0.615 bpw** |
| B: + diagonal-Hessian `pre` weighting and a post refit | 2.970 | 6.4977 | the weighting: **+0.053 PPL at the same size** |
| published NariQ 3.0bpw | 3.012 | 6.3523 | |

So coding the head is a good trade, and **the diagonal-Hessian weighting through `pre` costs real
quality even though it improves the quantity it optimises**: per module it cut the
activation-weighted error by 10-60% (section 9's sweep), and end to end that bought nothing and lost
0.05 PPL.  Two reasons it can go that way, and they point the same direction:

* the diagonal is not the Hessian.  `tr(dW H dW^T)` with the off-diagonal terms kept is what LDLQ /
  GPTQ-style sequential quantization minimises, and exllamav3 does exactly that; a per-column rescale
  can only match the diagonal, and any error it leaves correlated across columns is invisible to it.
* scaling columns before the Hadamard *reduces* incoherence, which is the property the trellis needs.
  A column scaled 16x is a structure in the rotated matrix, and the Viterbi has to spend bits on it:
  the Frobenius error rose from 0.132 to 0.137-0.166 across modules, which is the price being paid.

The reference's own `|pre|` is flat to +-2.5% and uncorrelated with anything we could measure
(section 6).  On this evidence that is not an oversight.  The weighting stays available behind
`--imatrix` because the mechanism is right and the proxy may be worth it on other objectives, but it
is not a default, and the honest next step is a full-Hessian sequential quantizer rather than a better
exponent.

## 11. What closed the gap: the frequency-banded head

The published file bands its output head by token frequency and gives the first band 6 bits, then 4,
then 3, then 2 for the rest.  Quantizing only that head and grafting it onto run A's blocks isolates
what the banding is worth, against the same blocks with a uniform 3-bit head:

| head | bits/weight | PPL | head cost vs BF16 head |
| --- | ---: | --- | --- |
| BF16 (run A) | 3.583 | 6.3686 | - |
| uniform 3-bit | 2.969 | 6.4448 | +0.0762 |
| **16 bands, 6/4/3/2** | **2.943** | **6.3774** | **+0.0088** |

The banded head is both *smaller* and nine times cheaper in perplexity: 0.36 GiB against 0.44, because
twelve of its sixteen bands are 2-bit, and the tokens the model actually emits keep 6 and 4 bits.  This
is the one thing the reference does that mattered, and it is a property of the vocabulary's frequency
distribution rather than of the weights -- which is why no amount of per-module work on the blocks
found it.

It runs: on one A100, `llama-simple -ngl 99` decodes 64 tokens at **40.9 tok/s** from the 2.943 bpw
file, and the prompt "The three laws of robotics are:" produces Asimov's three laws verbatim.  On the
CPU build the same file runs at 0.41 tok/s (the coded embedding is a CPU path by design).

Where that leaves the comparison.  Every row is the fork's own `llama-perplexity` over wikitext-2
(ctx 2048, 32 chunks) and its own `test-trellis-model` + `compare_logits.py` for the KL (65 logit rows,
the candidate teacher-forced on the reference's tokens):

| model | bytes | bits/weight | PPL | mean KL vs BF16 | top-1 |
| --- | ---: | ---: | --- | ---: | ---: |
| BF16 | 53,808,276,736 | 16 | 6.1963 +- 0.0831 | - | - |
| published NariQ 3.0bpw | 10,123,626,592 | 3.010 | 6.3523 +- 0.0835 | 0.1089 | 86.2% |
| **ours: 3-bit blocks, 2-bit embedding, 16-band head** | 9,895,557,568 | **2.943** | **6.3774 +- 0.0863** | 0.0839 | 87.7% |
| ours: 3-bit blocks, BF16 head | 12,048,789,056 | 3.583 | 6.3686 +- 0.0861 | 0.0681 | 90.8% |
| ours: 3-bit blocks, uniform 3-bit head | 9,983,781,216 | 2.969 | 6.4448 | - | - |
| ours: + per-module allocation (no weighting) | 10,142,685,536 | 3.016 | 6.4940 +- 0.0852 | 0.1801 | 86.2% |
| ours: + diagonal-Hessian weighting | 9,983,781,280 | 2.969 | 6.4977 +- 0.0867 | 0.0880 | 92.3% |
| ours: + weighting + allocation | 10,142,685,536 | 3.016 | 6.5253 +- 0.0854 | 0.2191 | 86.2% |

0.025 perplexity behind the reference at 0.067 fewer bits per weight, with a *uniform* bitrate over the
blocks.  Both attempts to do better than uniform lost, and they lost for the same reason.

## 12. Why both allocations failed: the imatrix is only half the sensitivity

The allocation is driven by `S_m = sum_j h_j ||W_m[:,j]||^2` and the weighting by `h_j` itself, and `h`
is the mean square of a module's *input* channels.  That makes `tr(dW H dW^T)` the loss only if the
module's output error reaches the loss with unit weighting, and it does not: an error in `ffn_down`'s
output lands directly in the residual stream, while an error in `in_proj_a`'s output passes through a
DeltaNet gate first.  Worse, `h` is not comparable across modules that read different spaces -- here
the hidden state's channel power is 0.34 to 198 while the FFN intermediate's is 3.3e-6 to 9.2e-3, four
orders of magnitude apart for no reason that has anything to do with how much either matters.

So the allocator, told only the input side, took bits away from exactly the matrices that carry the
model (2 bits for most of `ffn_down`, `out_proj` and `gate_proj`) and gave them to the 48x5120 DeltaNet
gates, whose total error contribution is negligible however well they are coded.  Doubling the FFN's
relative error from 0.132 to 0.262 is what the 0.049 perplexity costs.

The fix is not a better allocator but a sensitivity that includes the output side: the Gauss-Newton /
full-Hessian estimate a sequential quantizer produces as a by-product (LDLQ, as exllamav3 does), or a
direct measurement of each module's contribution to the loss.  Everything here is ready for either --
`bit_plan.allocate` takes any per-module error and energy, and `export.py --plan` takes any allocation.
## Appendix: the artefacts and the commands that made them

Everything lives on /mnt/ssd (the root filesystem on this box has ~1 GiB free).  Clones:
`/mnt/ssd/git/nari-llama.cpp` at 8e0e016 with `build-cpu` (the oracle) and `build-cuda` (sm_80,
`llama-perplexity`, `test-trellis-model`), `/mnt/ssd/git/exllamav3` at 8c593e4.

| artefact | what it is |
| --- | --- |
| `/mnt/ssd/nariq-ref/Qwen3.8-27B-NariQ-3.0bpw.gguf` | the published reference, sha256 matching `nariq/run.sh`'s pin |
| `/mnt/ssd/nariq-work/qwen38-27b-bf16.gguf` | the BF16 baseline, 51 GiB, from the same checkpoint |
| `/mnt/ssd/nariq-work/imatrix-wiki.gguf` | `llama-imatrix`, 64 chunks of wikitext-2 train, 400 tensors |
| `/mnt/ssd/nariq-work/vocab-order.npy` | token ids in frequency order on the same corpus |
| `/mnt/ssd/nariq-work/energies.json` | S_m per module, for the allocator |
| `/mnt/ssd/nariq-work/q27b-3bit{,.gguf}` | run A: uniform 3-bit, 2-bit embedding, BF16 head |
| `/mnt/ssd/nariq-work/q27b-3bit-cal{,.gguf}` | run B: + imatrix weighting, post refit, 3-bit coded head |
| `/mnt/ssd/nariq-work/q27b-graft-head{,.gguf}` | A's blocks with B's head, for the ablation |
| `/mnt/ssd/nariq-work/q27b-alloc`, `q27b-alloc-plain` | the allocated runs, with and without the weighting |

```sh
# the baselines
python convert_hf_to_gguf.py /mnt/ssd/models/Qwen3.8-27B --outfile bf16.gguf --outtype bf16 --no-mtp
python -m nariq.tools.measure ppl bf16.gguf --ngl 40 --chunks 32          # 6.1963 +- 0.0831
python -m nariq.tools.measure ppl Qwen3.8-27B-NariQ-3.0bpw.gguf --chunks 32   # 6.3523 +- 0.0835

# run A, and its measurements
CUDA_VISIBLE_DEVICES=4 python -m nariq.export /mnt/ssd/models/Qwen3.8-27B q27b-3bit \
    --bits 3 --embed-bits 2 --fill 0.95 --verify sample
python -m nariq.tools.evaluate q27b-3bit q27b-3bit.gguf --reference bf16.gguf --skip-dense
python -m nariq.tools.roundtrip_check q27b-3bit q27b-3bit.gguf --oracle --skip-embed --modules 6

# the ablation
python -m nariq.tools.graft q27b-3bit q27b-3bit-cal q27b-graft-head --take lm_head
python -m nariq.tools.evaluate q27b-graft-head q27b-graft-head.gguf --skip-dense

# the allocation
python -m nariq.tools.bit_plan /mnt/ssd/models/Qwen3.8-27B --imatrix imatrix-wiki.gguf \
    --errors q27b-3bit/nariq-export.json --energies energies.json --target-bpw 3.0 --out plan-3.0.json
CUDA_VISIBLE_DEVICES=6 python -m nariq.export /mnt/ssd/models/Qwen3.8-27B q27b-alloc-plain \
    --bits 3 --embed-bits 2 --head-bits 3 --plan plan-3.0.json --fill 0.95
```

A whole-model run is ~50 minutes at a uniform 3 bits (71 with the refit pass and a coded head) on one
A100 and never needs more than about 12 GiB of device memory.
