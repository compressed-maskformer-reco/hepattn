# Study: CLIC "Glow" Jet-Energy IQR discrepancy vs the paper

**Status:** workflow investigation CLOSED — no data/plotting bug. High-E IQR excess is real,
universal across all 4 runs (§5), and confirmed against the paper (§6): paper Glow's jet-E IQR
**falls** with energy (~0.05 at 200 GeV), ours **rises** (~0.10) — inverted trend, ~2× worse at
high E. Leading cause: a **model/config difference** (paper 12M params vs our 10.1M — §6.2).
Open work is training/paper-side only (§7).
**Started:** 2026-07-06
**Owner:** Maria Mazza
**Paper:** GLOW, arXiv:2508.20092 (linked in `../README.md`)

---

## 1. Goal

Understand why the **Jet energy-response IQR** produced from our trained CLIC particle-flow
("Glow"/`mpflow`) model does **not** match the published paper figure.

- **Symptom:** the Jet-E-response **median** looks correct (flat, centered near 0), but the
  **IQR rises to ~0.10 at high jet truth energy**, whereas in the paper it stays low and
  outperforms Pandora across the full energy range.
- **Question posed:** is this a corrupted input sample, a bug in our eval/plotting workflow,
  or something real about the model?

The plots are made with
`src/hepattn/experiments/clic/notebooks/performance.ipynb` (cells 4–16; `plot_jet_response`
is the disputed panel).

---

## 2. Which model produced the plots

The notebook's active "HiPerGator Paths" config (cell 4) reads:

```
logs/clic_v6_20260605-T113014/ckpts/epoch=195-val_loss=3.78800__test.root
```

→ the **3× L4 (1 node)** run, SLURM job 33954040, best checkpoint **epoch 195**
(`val_loss 3.78800`). This is the only run whose checkpoints have stored eval outputs
(`.h5` + `.root`). See `TRAINING_LOG.md` / `training_runs_report.md` for the run inventory.

**Important:** of all five runs, this one has the **lowest val_loss**:

| Run | best val_loss |
|---|---|
| **3× L4 (plotted)** | **3.788** |
| 1× B200 | 3.836 |
| 6× L4 (2 nodes) | 3.922 |
| 4× B200 (resubmit) | 3.975 |
| 4× B200 (original) | OOM, no ckpts |

So the plotted run is our **best-converged** model; evaluating the others is unlikely to
*improve* the IQR.

---

## 3. Reproduced symptom (ground truth for this study)

Full jet pipeline (genkt R=0.7, Hungarian ΔR<0.1, `dr_cut=0.1`, `leading_n_jets=2`,
`pt_min=10`), IQR = `p75 − p25` of `e_rel = (reco_E − truth_E)/truth_E`, binned in truth jet E:

| Jet E bin [GeV] | N | IQR |
|---|---|---|
| 0–20 | 1626 | 0.075 |
| 60–80 | 2657 | 0.077 |
| 100–120 | 2461 | 0.085 |
| 160–180 | 4144 | **0.099** |
| 180–200 | 4893 | **0.096** |

The IQR climbing toward ~0.10 at high E is **real and reproducible** from the stored
predictions.

---

## 4. What has been tried — hypotheses and verdicts

All code-level suspects were investigated (three parallel deep-dive agents + direct
inspection of the ROOT files). **Every one was ruled out.**

### 4.1 `reader.py` indicator-threshold change — RULED OUT
`load_pred_mpflow` was recently refactored (see `git diff` on
`performance/reader.py`) to:
```python
pred_ind = ak.to_numpy(...).astype(bool)
mask = pred_ind > threshold
```
Hypothesis: `.astype(bool)` collapses float probabilities to True, silently disabling the
`ind_threshold: 0.65` cut → too many particles → energy tail.

**Verdict: no effect.** In the eval ROOT file, `pred_ind` is stored as a **boolean**
(dtype bool, mean 0.322), because the indicator is already discretized at *write* time.
Old logic (`raw>0.65`) and new logic (`astype(bool)>0.65`) keep **identical** objects —
48.335/event in both, difference = 0. The notebook's `ind_threshold` is inert (only the
boolean, not the raw probability, is written to ROOT).

### 4.2 Input corruption / truncation & truth–pred misalignment — RULED OUT
- `test_clic_common_infer.root` (model input): 19,998 events, all branches read cleanly —
  **not truncated**. (The earlier truncation incident was only `val_clic_fix.root`.)
- Truth file `test_clic_common_raw.root`: 20,000 events, **no `event_number` branch**.
- Alignment is **provably correct**: the truth loader drops the two track-and-topocluster-less
  events (raw indices 1399, 11288) and re-indexes positionally; the infer file is exactly
  `raw minus {1399, 11288}` in original order; both use consistent positional IDs; the
  notebook does a real `event_number` intersection+reorder
  (`performance.py:reorder_and_find_intersection`). Every prediction is paired with the
  correct truth jet. The 19,998→19,722 drop is the dataset's ">150 particles / too many
  nodes" cap — expected.

### 4.3 Neutral-pt "no-op" write bug — REAL BUG, but PROVEN IMMATERIAL here
`predictionwriter.py:52` (original upstream code, PR #34) intends to set neutral particles'
`pt = E/cosh(eta)` but was a numpy no-op:
```python
pflow_ptetaphi[neutral_mask][..., 0] = ...   # assigns to a COPY → discarded
```
So neutrals kept their directly-regressed pt instead of the calorimeter-energy-derived pt.
This *looked* like the perfect culprit (neutral-dominated high-E jets, resolution effect).

**Fix applied** (single advanced-index assignment, writes in place):
```python
pflow_ptetaphi[neutral_mask, 0] = pflow_data[neutral_mask, 0] / np.cosh(pflow_ptetaphi[neutral_mask, 1])
```

**Verdict: immaterial for this model.** The model regresses `e` and `pt` *consistently*
(`pred_pt ≈ pred_e/cosh(eta)`, the massless relation), so the correction shifts neutral pt
by **<1%** (`new/old ≈ 1.00` across all η; kept-neutral ⟨pt⟩ 2.197→2.206). Re-running the
full jet pipeline on the fixed file gives an **identical** IQR curve:
**max |ΔIQR| = 0.0012** (statistical noise). See `jet_iqr_old_vs_fixed.png`.
The fix is kept because it is a legitimate correctness improvement for future models, but it
is **not** the cause of the discrepancy.

### 4.4 L4 precision fix & evaluation precision — RULED OUT
Two precision concerns were raised: (a) did the L4 FlashAttention fix hurt training? (b) did
we miss a precision option at eval time?

- **Eval precision is already maximal.** `configs/eval.yaml` sets `precision: 32-true`
  (full fp32), `attn_type: torch` (standard, non-flash attention), `matmul_precision: highest`,
  `is_inference: true`. This matches the CLIC `README.md` eval instructions **exactly** — those
  four flags are precisely what the README lists as *required* for correct evaluation. Nothing
  was missed; `32-true` *is* the full-precision option. The L4 fix lives in the flash-varlen
  path, which eval does not execute (`attn_type: torch`).
- **The L4 fix restores the intended training precision, it doesn't lower it.** The fix
  (`attention.py`, guarded by `if q_flat.dtype == torch.float32`) casts flash-attn inputs to
  **bf16** — exactly the precision `base.yaml` asks for (`precision: bf16-mixed`). On L4,
  `torch.compile` broke autocast so the kernel silently received fp32; the fix realigns it with
  the recipe. bf16-mixed keeps fp32 master weights (standard mixed precision).
- **Decisive evidence:** the high-E IQR climb reproduces **identically on the B200 runs**
  (different hardware, native bf16 autocast, fix is a no-op there). A "clean" bf16 training on
  flagship GPUs gives the same result → the L4 precision handling is not the differentiator.

The only precision-related unknown is whether the *paper* trained at a different precision than
the repo's own `bf16-mixed` — that is a paper-recipe question (§6), not our L4 fix or eval.

### 4.5 Net conclusion of Phase 1
The IQR shape is a genuine property of the model's predictions, **not** a data or workflow
bug. This points to a **model-vs-paper training gap** (recipe, convergence, or a differently-
trained/config'd model) rather than a plotting error — now substantiated by the paper
comparison in §6.

---

## 5. Cross-run eval comparison — DONE (result below)

Evaluated the *other* trained runs and overlaid their Jet IQR curves against Pandora, to test
whether "IQR → 0.10 at high E" is **universal across all our runs** or specific to the 3× L4 run.

Ran as SLURM eval jobs (`submit_eval_run.sh`, best ckpt of each run) → all completed:

| Run | Job | Status | Output |
|---|---|---|---|
| 1× B200 (T000306) | 36445950 | ✅ done (6:56) | `…/ckpts/epoch=199-val_loss=3.83626__test.{h5,root}` |
| 4× B200 (T102707) | 36445952 | ✅ done (7:00) | `…/ckpts/epoch=190-val_loss=3.97507__test.{h5,root}` |
| 6× L4 (T143447) | 36445951 → **36446174** | ✅ done (1:50) | `…/ckpts/epoch=199-val_loss=3.92173__test.{h5,root}` |

> **Gotcha fixed:** the 6× L4 run trained on 2 nodes, so its frozen `config.yaml` has
> `num_nodes: 2`; Lightning aborts eval when SLURM `--nodes=1` doesn't match (job 36445951
> died in 29 s with `ValueError: num_nodes=2 ... does not match --nodes=1`). Fixed by adding
> `--trainer.num_nodes=1` to `submit_eval_run.sh` and resubmitting as job 36446174.

**Result (see `jet_iqr_all_runs.png`) — the IQR shape is UNIVERSAL across all four runs:**

| Jet E [GeV] | 3× L4 | 1× B200 | 6× L4 | 4× B200 | **Pandora** |
|---|---|---|---|---|---|
| ~10  | 0.075 | 0.076 | 0.074 | 0.075 | 0.091 |
| ~90  | 0.077 | 0.078 | 0.081 | 0.081 | 0.061 |
| ~170 | 0.099 | 0.097 | 0.111 | 0.096 | **0.056** |
| ~190 | 0.096 | 0.098 | 0.111 | 0.098 | **0.054** |

Every run beats Pandora at **low** jet E but is **worse** than Pandora at **high** jet E, with
IQR climbing to ~0.10, independent of hardware / batch size / val_loss. The high-E IQR excess
is therefore **inherent to this repo's model + config + data**, not a fluke of one run and not
a workflow bug (all code-level causes were closed in §4).

**Interpretation:** the discrepancy is real and inherent to this repo's model, not a plotting
bug. See §6 for the direct comparison to the paper, which pins down *what* is different.

---

## 6. What the paper actually shows (arXiv:2508.20092)

Read the GLOW paper (linked in `README.md`) and its §5.2 "Jet-level performance" (Figures 3–4).
This **corrects the original premise and confirms the discrepancy is real**:

**6.1 The premise was slightly off — but the discrepancy is real and worse than thought.**
- The user recalled "Glow should beat Pandora across the full jet-E range." In the paper,
  **Figure 4** (median + IQR of jet-E relative residual **vs truth jet energy**) compares
  **Glow to HGPflow only — Pandora is NOT in Figure 4.** Glow-beats-Pandora is a *Figure 3*
  statement (overall residual distributions: "Both Glow and HGPflow outperform MLPF and
  Pandora"). So the baseline in the energy-dependent plot is HGPflow, not Pandora.
- **The real problem is the trend, not the baseline.** In the paper's Figure 4, **Glow's IQR
  DECREASES with jet energy** — from ~0.08–0.10 at ~20 GeV down to ~**0.04–0.05 at ~200 GeV**
  (y-axis ~0–0.12, x-axis ~20–200 GeV). The paper: *"Glow maintains the best median accuracy
  (within ±2%) and consistently achieves a relative improvement of 15% in jet energy resolution
  over HGPflow across all energy ranges."*
- **Our models do the OPPOSITE:** IQR **RISES** from ~0.075 (low E) to ~0.10 (high E) — see §5.
  So at high E our resolution is **~2× worse than the paper's Glow** and the energy trend is
  **inverted**. Falling relative resolution with energy is the physically expected behaviour
  (stochastic term ∝ 1/√E; Pandora shows it too: 0.091→0.054). Our rising trend is the anomaly.
- Net: the user's instinct ("the IQR plot looks wrong") was **correct** — just misattributed to
  the Pandora baseline. The genuine, confirmed discrepancy is **rising vs falling IQR with jet
  energy**, ~2× at the top of the range.

**6.2 Concrete config differences vs the paper (leading leads):**
| | Paper (arXiv:2508.20092) | Our runs |
|---|---|---|
| Parameters | **"Glow has 12M parameters"** | **10,126,115 (~10.1M)** (`TRAINING_LOG.md`) |
| Hardware | 2× H100 | L4 / B200 (various) |
| Batch size | **1024** | 768 / 1536 / 2048 / 8192 |
| Epochs | 200 | 200 |
| Train time | 23 h (Isambard-AI) | — |
| Precision | not stated in paper | `bf16-mixed` (repo `base.yaml`) |

> **Strongest lead: the parameter count.** The paper reports **12M** params; every run here has
> **10.1M** (per `TRAINING_LOG.md`). 10.1M does not round to 12M — this suggests the current
> repo `base.yaml` may be a **smaller / drifted model** than the one that produced the paper's
> Figure 4. A reduced-capacity model plausibly loses high-energy jet resolution first (dense,
> high-multiplicity jets), which matches the inverted trend. **Verify the architecture in
> `base.yaml` against the paper's Appendix/config before anything else.**

---

## 7. Where to start if you pick this up

1. **FIRST: reconcile the parameter count (§6.2).** Paper = 12M, our runs = 10.1M. Diff the
   current `configs/base.yaml` model architecture (dims, depth, num heads/layers, num queries)
   against the paper's Appendix / the upstream `samvanstroud/hepattn` config at the paper's
   commit. If the repo config drifted to a smaller model, that is the most likely cause of the
   inverted high-E IQR trend — retrain with the paper's architecture and re-evaluate.

2. **Get the paper author's reference prediction file** (`sam_clic_best.root`, referenced only
   in the notebook's WIS config; not present locally). Running it through `compare_runs_iqr.py`
   gives a *direct* apples-to-apples IQR overlay of paper-Glow vs our runs — the single most
   decisive check that our pipeline reproduces their number for their predictions.

3. **Convergence / recipe:** paper used batch **1024** on 2× H100; ours 768–8192. All our runs
   already show the same inverted trend, so batch within that range is unlikely to be the whole
   story — but a paper-matched batch/LR/optimizer run is worth one attempt. (Paper doesn't state
   optimizer/LR/precision; ours is Lion + bf16-mixed.)

4. **High-E jet resolution mechanism** (if the model turns out to match the paper): decompose
   the high-E IQR tail — charged vs neutral jets? merged/split particles in dense jets? the
   `num_objects=150` query cap biting on busy high-E events? Use
   `perf.data[name]["jet_residuals"]` and split by jet constituent type.

5. **Do NOT re-investigate** §4.1–4.4 — closed with direct evidence (reader, inputs, neutral-pt,
   precision/eval flags all ruled out).

---

## 8. File & artifact index

**Source (repo, committed/tracked as noted):**
- `notebooks/performance.ipynb` — plotting notebook; cell 4 = active config; cell 16 = IQR plot.
- `performance/reader.py` — `load_pred_mpflow` (§4.1); `load_truth_clic` (§4.2 alignment).
- `performance/performance.py` — pipeline: `reorder_and_find_intersection`, `compute_jets`,
  `hung_match_jets`, `compute_jet_res_features`.
- `performance/plot_helper_event.py:222` — `plot_jet_response` (IQR = p75−p25 of `e_rel`).
- `predictionwriter.py:52` — **neutral-pt fix applied here** (§4.3).
- `pflow_data.py:118` — positional `event_number`; `:120` — >150-particle event drop.
- `submit_eval_run.sh` — parameterized eval submission (with the `num_nodes=1` fix).

**Eval outputs (each run's `logs/<run>/ckpts/`):**
- 3× L4: `epoch=195-val_loss=3.78800__test.{h5,root}` (original, plotted)
- 3× L4: `epoch=195-val_loss=3.78800__test_neutralptfix.root` (fixed neutral pt; IQR identical)
- 1× B200: `epoch=199-val_loss=3.83626__test.{h5,root}`
- 4× B200: `epoch=190-val_loss=3.97507__test.{h5,root}`
- 6× L4: `epoch=199-val_loss=3.92173__test.{h5,root}`

**Analysis scripts (co-located in this `studies/` directory):**
- `reconvert_neutralptfix.py` — regenerate a run's ROOT from its existing `.h5` with the
  neutral-pt fix, to a new `__test_neutralptfix.root` (never overwrites).
- `prove_iqr_unchanged.py` — runs the pipeline on original vs fixed and overlays IQR
  (produced `jet_iqr_old_vs_fixed.png`, max |ΔIQR| = 0.0012).
- `compare_runs_iqr.py` — overlays all runs' IQR vs Pandora (produces `jet_iqr_all_runs.png`).
- `jet_iqr_old_vs_fixed.png` — the §4.3 proof plot.

> Note: each script writes its output PNG to a hard-coded `OUT` path (the original session
> scratchpad). Before re-running, edit `OUT` at the top of the script to a directory you
> control (e.g. this `studies/` dir). Run them in the `clic` pixi env (`pixi run -e clic python …`).

**Run inventory:** `../../../TRAINING_LOG.md`, `../../b200_utilization/training_runs_report.md`.
