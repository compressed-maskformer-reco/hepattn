# Linformer vs quadratic attention — float comparison for FastML

**Owner:** Erdem Ertorer · **Started:** 2026-08-26 · **Branch:** `linformer-float`
**Goal:** a jet-E median/IQR curve for Linformer, overlaid on the reference MaskFormer
curves, on identical code and identical evaluation.

## Base selection (done)

| fork | vs `lgray/main` (1df05cc) | verdict |
|---|---|---|
| `asgrover-cmu/main` (Akum) | **5 behind**, 0 ahead, last push 2026-05-14 | wrong base — stale, no changes of its own |
| `mmfsz/main` (Maria) @ 6ee4c60 | **5 ahead** | **chosen** — carries `configs/eval.yaml`, RNTuple-capable `performance/reader.py`, and `studies/glow_jet_iqr/` |
| `helenllii/linformer-hardmask` @ e207a434 | 9 ahead / 5 behind | source of the Linformer |

Linformer exists **only** on Helen's branch and our `keras-hgq2`. Our torch
`linformer.py` is semantically identical to hers (token-level diff: one unused import,
`0.` vs `0.0`).

Ported: `models/linformer.py`, `configs/linformer.yaml` (verbatim), `models/attention.py`
(3-way merge — Helen branched before lgray/main, so her edits replay over Maria's tree;
**both** her linformer branch and Maria's bf16 cast survive), `models/__init__.py` export.
`encoder.py` skipped (blank line only).

## Measured on this tree (both configs built through the real CLI)

```
configs/base.yaml       10,126,115 params  (10.13M)   attn: 17 torch + 7 flash-varlen
configs/linformer.yaml  12,471,587 params  (12.47M)   attn: 24 linformer
```

The baseline reproducing Maria's documented **10.1M exactly** is the evidence the merge
left the quadratic path untouched. `tests/models/` failure set is byte-identical before
and after the port (6 pre-existing macOS inductor / flash-attn failures).

## ⚠️ Two findings that constrain the physics claim

### 1. Masked Linformer CANNOT compress: `k >= sequence length` is required

The attention mask is built in the **projected** space (last dim `k`) but filled from an
original-sequence mask (last dim `kv_len`), so `k < kv_len` raises. Measured boundary at
kv_len=168:

| k | 64 | 128 | 167 | 168 | 200 | 256 |
|---|---|---|---|---|---|---|
| masked | CRASH | CRASH | CRASH | OK | OK | OK |
| unmasked | OK | OK | OK | OK | OK | OK |

The CLIC config leaves `linformer_proj_dim` unset → **k=256 default, against 168 hits**.
So the "compression" matrix is an **expansion**, and it is *forced by the mask handling*,
not chosen. The decoder always masks (MaskFormer mask-attention, `decoder.py:273`), so the
decoder can never compress as implemented. The encoder only masks if a `mask_mod`/window
is configured — an unmasked encoder *could* take k < n.

**Consequence: this configuration is not an efficiency win.** It is +23% parameters and a
larger attention inner dimension. Any FastML claim must be about *accuracy at fixed
mechanism*, or the mask handling must be fixed first.

### 1b. Linformer is parameter-ADDITIVE, and at n=168 it cannot save compute either

Measured parameter buckets (both configs built through the real CLI):

| bucket | quadratic | Linformer | delta |
|---|---|---|---|
| attn Q/K/V/O projections | 4,737,024 | 4,723,200 | -13,824 (bias only) |
| attn E/F projection | 0 | 2,359,296 | **+2,359,296** |
| MLP / Dense | 5,291,219 | 5,291,219 | **0** |
| norms | 36,864 | 36,864 | 0 |
| other | 61,008 | 61,008 | 0 |
| total | 10,126,115 | 12,471,587 | +2,345,472 |

W^Q/W^K/W^V/W^O are `dim x dim` — their size does not depend on sequence length, so
compressing n cannot shrink them. Linformer shrinks the n x n attention matrix, which is an
*activation*, not a parameter. E/F are extra weights on top:

    params(Linformer) = params(standard) + 2 * seq_len * k   per attention module

Measured: 2*256*256 = 131,072 x **18** modules (6 encoder + 12 decoder) = 2,359,296 = the
entire delta. **It is not the MLPs** (byte-identical) and **not a small head_dim** (the
Q/K/V/O block is unchanged).

FLOPs at n=168, d=256, B=8 (analytic and measured agree exactly):

```
attention core (n^2) : 0.231 GFLOP = 4*B*n^2*d    <- only 25%
Q/K/V/O proj (d^2)   : 0.705 GFLOP = 8*B*n*d^2    <- 75%
```

| k | GFLOP | vs quadratic |
|---|---|---|
| 32 | 0.793 | 0.85x (best case) |
| 84 | 0.936 | **1.00x — break-even at k = n/2** |
| 168 | 1.167 | 1.25x (minimum the masked path allows) |
| 256 | 1.409 | **1.51x (our config)** |

Core fraction is `n/(2d+n)`; with d=256 you need **n >~ 512** before the quadratic term is
even half the cost. **Break-even k=84 is below the minimum k=168 the mask requires, so on
this model Linformer can never be cheaper than quadratic attention.**

> Measurement trap: `torch.utils.flop_counter.FlopCounterMode` does NOT count SDPA's
> attention core, but DOES count Linformer's explicit einsums. Comparing them naively
> understates quadratic by 25%. Force the core to explicit matmuls before comparing.

**Framing consequence:** an efficiency claim is unsupported at CLIC's n=168. The honest
claim is accuracy at fixed mechanism; the efficiency argument belongs at large n
(HL-LHC-scale hit multiplicities), which is where Linformer was designed to pay off.

### 2. Linformer's 12.47M is dangerously close to the paper's 12.1M

Akum's reference plot has "Paper MaskFormer 12.1M" and "Maskformer 10.1M". Our Linformer
lands at 12.47M for an unrelated reason (the E matrices). **Label it explicitly** or the
curves will be misread as a parameter-matched comparison.

## Comparability requirements (from Maria's `glow_jet_iqr/`)

- Overlay tool: `studies/glow_jet_iqr/00_cross_run/compare_runs_iqr.py`
- Convention in that tool: `network_type: mpflow`, **`ind_threshold: 0.65`**,
  truth `test_clic_common_raw.root`, jets `dr_cut=0.1, leading_n_jets=2, pt_min=10`,
  IQR = p75−p25 of `e_rel` in 20 GeV bins of `ref_e`.
- Eval must use `configs/eval.yaml` (`precision: 32-true`, `matmul_precision: highest`,
  `is_inference: true`).

> ### ⚠️ `eval.yaml` will silently destroy a Linformer evaluation
> It sets `model.model.encoder.attn_type: torch`, rebuilding the **encoder** as standard
> attention. Measured: the two modules share **zero** state-dict keys —
> linformer has `attn.{proj_k,proj_v,to_q,to_k,to_v,to_out}`, torch has
> `{in_proj_weight,in_proj_bias,out_proj.*}`. Loading a Linformer checkpoint gives
> `missing=4 unexpected=7`: a strict load raises, a non-strict load leaves the encoder
> **randomly initialised** and the run produces plausible-looking garbage.
> Override it back with `--model.model.encoder.attn_type=linformer` (the decoder is not
> touched by eval.yaml and stays linformer either way).

## The run: configs/linformer_polaris.yaml

Generated from v6 `base.yaml` (the config behind the 10.1M reference curves) by
`mkcfg` — regenerate rather than hand-edit, so provenance stays checkable.

| change | v6 | ours | why |
|---|---|---|---|
| encoder `attn_type` | flash-varlen | **linformer** | the thing under test |
| decoder `attn_type` | (absent → torch) | **linformer** | the thing under test |
| `optimizer` | Lion | **AdamW** | Lion reportedly does not converge with Linformer (Erdem, measured) |
| `lrs_config.max` | 8e-5 | **1e-4** | Lion is sign-based; its LR does not transfer. 1e-4/0.03 is THIS repo's last validated AdamW recipe for the CLIC MaskFormer (`base.yaml` @ 22bb033, 2025-07-28, immediately before the Lion switch) |
| `weight_decay` | 1e-4 | **0.03** | pairs with the AdamW LR above; Lion's 1e-4 would be near-zero regularisation for AdamW |
| `batch_size` | 2048 | **256** | v6's 2048/GPU was tuned for a 192 GB B200. Maria's 24 GB L4 ran 256/GPU; 256 x 4 A100-40GB = global 1024, same family as the PLOTTED baseline (global 768) |
| `num_workers` | 16 | 8 | 32 cores / 4 ranks |
| paths | cmsuf | `/eagle/<project>/clic` | Polaris; `test_path` -> val file (the infer file has -9999 sentinels) |

Unchanged: architecture, 200 epochs, bf16-mixed, devices 4, scaling dict, OneCycle
schedule shape. Verified through the real CLI: 12,471,587 params, all 24 attention
modules linformer, AdamW instantiated with wd 0.03 / max_lr 1e-4 and **all** trainable
params in the optimizer.

#### Polaris bring-up: three failures, all environmental

1. **`.venv/bin/activate` not found.** venv lives under `WORK_ROOT` on /eagle; the branch
   is checked out in $HOME. hepattn is NOT installed into the venv, so PYTHONPATH decides
   which clone runs -- and the /eagle clone is on keras-hgq2. A preflight now asserts
   `hepattn.__file__` is inside `$REPO_DIR/src`.
2. **`nvc-Error: Unknown switch -Wno-psabi`.** Polaris puts NVHPC's `nvc` on PATH; Triton
   builds its CUDA helper with `$CC` and passes GCC-only flags. Fixed by exporting
   `CC=gcc` (7.5.0 on Polaris works). Not avoidable by config -- loss.py:352-359 wraps
   every cost function in torch.compile at import.
3. **`RuntimeError: shape '[1344, 256]' is invalid for input of size 64`** in the compiled
   BACKWARD. torch.compile (via the `Compile` callback, `dynamic=True`) generates broken
   inductor code for the linformer path on torch 2.10: the kernel allocates a 64-element
   workspace then views it as `[64 + 8*s27, 256]`. Forward and the validation sanity check
   pass; it dies on the first backward. **Dropped the `Compile` callback.** This is a
   deviation from the v6 baseline, which trained WITH it -- speed only, not numerics, but
   worth stating. loss.py's compiled cost functions are unaffected and still work.

Also a smoke-harness artifact worth remembering: OneCycleLR's first phase spans
`pct_start*total_steps - 1` steps, so a short `limit_train_batches` (<40 at pct_start
0.05) makes `get_lr()` divide by zero. Not a config fault.

## Confounds to state on any plot (ranked)

1. **Optimizer**: ours AdamW, the v6 baseline Lion. Unavoidable if Lion truly fails with
   Linformer, but it means the curve is not a pure attention ablation. Closing it needs an
   AdamW *quadratic* baseline — a second 200-epoch run.
2. **`value_residual` AND `qkv_norm` are both silently inactive under Linformer.** The
   linformer branch skips `_prepare_qkv`, where both are applied. v6 sets
   `value_residual: true` on the encoder, and `hybrid_norm: true` forces qkv_norm on for
   every attention module (`qkv_norm = qkv_norm or hybrid_norm`, norm.py). Measured on the
   real model: **48,208 parameters receive no gradient** — `value_residual_mix` on 5
   encoder layers (layer 0 is `is_first_layer`) plus `q/k/v_norm` on all 18 modules.

   This is not cosmetic: DDP aborts with *"parameters that were not used in producing the
   loss"*. Worked around with `strategy: ddp_find_unused_parameters_true`. It cannot be
   fixed by disabling `hybrid_norm`, which also drives `attn_norm`/`dense_post_norm` and
   would change the real architecture. **So the Linformer model is missing qkv-norm and
   the value residual relative to the quadratic baseline** — the largest architectural
   confound in this comparison. Proper fix: apply both inside `LinformerAttention`.
3. **Parameters**: 12.47M vs 10.13M (+23%), entirely the E/F matrices — see §1b. Do not
   present this as parameter-matched, and do not let 12.47M be confused with the paper's
   12.1M.

## Corrections to earlier assumptions

- The neutral-pT `predictionwriter.py` no-op **is a real bug but is immaterial here**:
  Maria measured max |ΔIQR| = 0.0012 and <1% shift in neutral pT, because the model
  regresses `e` and `pt` consistently (massless relation). It does **not** explain the
  neutral gap. Do not re-open (`glow_jet_iqr/00_cross_run/jet_iqr_discrepancy.md` §4.3).
- The paper-vs-HEAD IQR gap is **temporal drift, not fork drift**: HEAD is 100 commits
  past the `clic-paper` tag (`fb90390`). Leading suspect is the `Dense` refactor (#212)
  doubling the incidence-regression MLP width. So our Linformer curve is comparable to
  the **10.1M HEAD** baseline, not to the paper-tag 12.1M curve.

## Open — needs Akum

Which repo/commit and which branch (`mpflow_proxy` vs regression) produced his latest
plot. If it predates Maria's eval tooling the baselines need regenerating.

## 3. Where the `k >= n` constraint actually comes from (2026-08-29)

Measured with `Attention(attn_type="linformer")` at n=168, d=256, H=16.

### 3a. Only the DECODER is constrained. The encoder can take any k.

| k | encoder self-attn (`kv_mask` only) | decoder cross-attn (`attn_mask`) |
|---|---|---|
| 2, 8, 32, 84, 128, 167 | OK | **RuntimeError** (expand 2/8/.../167 vs 168) |
| 168, 256 | OK | OK |

The encoder never passes an `attn_mask` — its only mask is `kv_mask` (padding), which
`linformer.py` now handles by zeroing rows *before* the sequence projection. That path
has no k constraint at all. So **an encoder-only Linformer at k << n is available today**,
no code change required; the config just never sets `linformer_proj_dim`, so it inherits
the lucidrains default of 256.

### 3b. The mask that forces `k >= n` does not implement mask-attention

`linformer.py` builds the mask in the *projected* space and fills it from an
original-sequence mask:

```python
mask[..., :kv_len] = original_mask   # <- treats projected column j as token j
mask[..., kv_len:] = True
```

Projected column j is a learned mixture of every original position, so there is no
correspondence to token j. Two measured consequences:

**(i) Masked hits still leak.** Forbid query 0 from seeing hit 5, then perturb only hit
5 and measure the change in query 0's output (contract: must be exactly 0):

| | query 0 (hit 5 masked) | query 1 (unmasked control) |
|---|---|---|
| standard attention | **0.000e+00** | 3.105e+00 |
| linformer k=168 | 7.934e-02 | 7.346e-02 |
| linformer k=256 | 2.298e-02 | 1.659e-02 |

The "masked" query moves as much as the unmasked one.

**(ii) k > n is actively harmful, not merely wasteful.** Columns `kv_len:` are masked
unconditionally, so whenever *any* attn_mask is present, 88 of the 256 projected columns
(34%) are discarded — for every query, including unmasked ones. Restricting query 0 while
leaving query 1 free changes query 1's output by:

| | k=168 | k=256 |
|---|---|---|
| seed 0 | 0.0% | 19.9% |
| seed 1 | 0.0% | 29.8% |
| seed 2 | 0.0% | 29.5% |

At k=168 there are no surplus columns and query 1 is correctly untouched.

### 3c. Is there a real lower limit on k?

Not from n. Linformer's own bound (Wang et al. 2020) is `k = O(d/eps^2)` — a
Johnson-Lindenstrauss argument, so it scales with the head dimension, **not** the
sequence length. Here `d_head = 256/16 = 16`. For comparison, the only published
transformer-on-FPGA result on this stack (Laatu, Sun et al., arXiv:2510.24784) runs
**k = 2** at n = 8..64 and still beats full MHA under a fixed EBOPs budget.

The real obstruction is structural and specific to MaskFormer: the decoder's
`attn_mask` is `(B, num_queries, num_hits)` — a *different* mask per query — while
Linformer projects K/V once, shared across all queries. A per-query key mask cannot be
expressed after that projection without a per-query projection, which is the n^2 cost
being avoided. So it is a genuine incompatibility, not a bug to patch.

**Options, in increasing order of cost:**
1. Linformer in the encoder only (k free, e.g. 32-84), quadratic in the decoder.
   Break-even is k = n/2 = 84, so k <= 84 is a real FLOP win. No code change.
2. Set `mask_attention: false` and use Linformer in both — changes the physics
   (MaskFormer's iterative mask refinement is a core mechanism), so it needs its own
   ablation before it can be called a Linformer-vs-quadratic comparison.
3. Fix the decoder mask properly (project the mask, or mask before projection per
   query). Only (3) preserves both mechanisms, and it is not obviously cheap.

**Correction to the poster's defect 3.** It says k=256 > n=168 "expanded the sequence
instead of compressing it". True, but incomplete: the surplus columns are then thrown
away by (ii), and the mask they were protecting does not work anyway.

## 4. Sorting the node sequence for Linformer (2026-09-10 .. 09-15)

Lindsey's suggestion: sort hits in phi so Linformer's position-indexed projection `E[n, k]`
has a geometric meaning. Quadratic attention is permutation-equivariant and the
`FourierPositionEncoder` acts on each node's own eta/phi, not its index, so sorting is a
strict no-op for the quadratic arm -- a clean control. Precedent: Sun et al.
(arXiv:2510.24784, sec. 2.1) sort constituents by pT and run k=2.

Implemented as a **data-reader** option, `data.sort_nodes_by: phi|eta|type_phi|type_eta`
(`pflow_data.py`, `node_sort_order`). One permutation is applied to every per-node tensor
AND to the incidence-matrix columns. Tests: `tests/experiments/clic/test_node_sort.py`
(15 tests; mutation-checked: no-incidence-permute fails 4, no-feature-permute fails 4,
identity order fails 8, dropped block key fails 1).

### 4a. Why not the built-in `hepattn.utils.sorter.Sorter`? (measured, not read)

It is wired into `MaskFormer(sorter=...)` and used by two trackml configs, but for CLIC it
misaligns truth and inputs. Demonstrated on one real event through `load_event`:

| tensor | moved by `Sorter`? | consumer |
|---|---|---|
| `node_valid` | yes | padding |
| `particle_node_valid` | yes | mask BCE/dice |
| `particle_incidence` | **no** (key lacks "node") | incidence KL, `task.py:1351,1366` |
| `x["inputs"]["node_e"/"node_pt"/...]` | **no** | regression proxy, `task.py:1490` |

After `sort_targets`, `particle_node_valid` and `particle_incidence > cut` disagree on 5 of
30 (query, node) cells over the 5 real nodes. So with the built-in sorter the mask head
trains in phi order, the incidence head trains in file order, and the proxy sums the
energies of the wrong nodes. Any CLIC run using `sorter:` has all three.

### 4b. Small-k encoder-only arm (Helen)

`link32_polaris.yaml` / `link32_sortphi_polaris.yaml`: encoder `linformer_proj_dim: 32`,
`linformer_seq_len: 168`, decoder back to `torch`. Measured 10,165,459 params vs quadratic
10,126,115 (within 0.4%) vs the k=256 arm's 12,451,027. Brief: `SMALL_K_EXPERIMENT.md`.
Completes a 2x2: {k=256, k=32} x {file order, phi-sorted}.

## 5. Quadratic arm: matched-epoch result (2026-09-15)

Run `clic_v6_quadratic_20260911-T042113` (job 7598123, `quadratic_polaris.yaml`, commit
6dbb565 tree; no sorting, no cos/sin fix -- same inputs as the Linformer arm). Died at
epoch 162 on `Disk quota exceeded` while copying the checkpoint from node-local $TMPDIR
into $HOME (51 GB used of 45 GB quota); the epoch-162 file is a 32 MiB truncation and was
deleted. Resumed from epoch 161 as job 7624470 (06:00 walltime, ~38 epochs).

Validation loss at matched epoch, same OneCycle schedule, from checkpoint filenames:

| epoch | quadratic | Linformer k=256 | gap |
|---|---|---|---|
| 160 | 4.00455 | 4.56788 | 0.563 |
| 161 | 4.01215 | 4.57743 | 0.565 |
| 199 | **3.98111** (resumed run `clic_v6_quadratic_20260915-T230910`, finished 2026-09-16) | 4.51461 | 0.534 |

The quadratic arm at epoch 161 is already 0.50 below the Linformer arm's *final* loss.
Epoch-to-epoch jitter on either arm is ~0.01, so this is not noise. The arms differ in
exactly four config lines (encoder attn_type, decoder attn_type, value_residual, name)
and the quadratic arm has 2.3M FEWER parameters.

Interpretation, pending jet IQR from the eval: consistent with sec. 3b and with Maria's
hypothesis (2026-09-15) that the poster's poor physics came from the non-functional
decoder mask under Linformer rather than from Linformer per se. Helen's encoder-only
k=32 arm (decoder quadratic, mask attention genuinely active) is the decomposition test:
if it lands near the quadratic arm, the decoder mask was the whole story.
