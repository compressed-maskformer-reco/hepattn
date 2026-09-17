# Task 4 — Fork diff vs upstream `samvanstroud/hepattn`

**Status:** ✅ DONE (2026-07-07) — **REFRAMES THE WHOLE INVESTIGATION**

## RESULT — it is NOT fork drift; it is TEMPORAL drift within upstream

Verified directly:
- **This fork's committed code is byte-identical to `upstream/main`**: `git rev-list --left-right
  --count HEAD...upstream/main` = `0  0`; `merge-base HEAD upstream/main` == `HEAD` (`1df05cc`).
  There is **no lgray-vs-samvanstroud drift** to find.
- The paper ("GLOW") was trained at the tag **`clic-paper` = `fb90390` (2025-08-11, "update clic
  #103")**. We train from **`HEAD`, which is 100 commits + a major model refactor LATER**
  (`git rev-list --count clic-paper..HEAD` = 100). **So we are not training the paper's model.**

### Ranked candidate causes (paper tag → HEAD)
1. **Incidence-regression MLP width doubled — SEMANTIC, active in CLIC config, VERIFIED.**
   `Dense` changed (#212 "simpler dense", `models/dense.py:36-37`): an `int` `hidden_layers` now
   expands to `[input_size * hidden_dim_scale] * n` (scale default **2**).
   - Paper tag `base.yaml`: incidence `net.hidden_layers: [256,256]`, `output_size: 256`;
     `node_net.hidden_layers: [256]`.
   - HEAD `base.yaml:209,214`: `hidden_layers: 2` → **[512,512]**, `1` → **[512]** (output_size
     now defaults to input=256). The incidence matrix distributes hit energy to particles
     (`task.py`: neutral energy = `inc_e_weighted.sum(-1)`), so a different-capacity incidence
     head directly changes the energy sharing that drives jet-E resolution. **Leading suspect.**
2. **Encoder/decoder norm + attention refactor — SEMANTIC, active by default.** HybridNorm made
   depth-dependent (#208/#226), qkv-norm threaded into `Attention`, query posenc added inside
   decoder layers (#141/#228), decoder layer restructured "following SAF" (#224).
3. **batch_size / LR recipe** (lower priority for the *plotted* run, which used global ~768,
   close to paper's 512×2=1024; the working-tree `batch 2048/devices 4` is a local edit, not
   what the plotted run used).

### 12M vs 10.1M param gap — NOT a config knob
Size-determining config (`dim 256`, enc 6L/16h/8reg, dec 4L/150q) is **identical** at both revs.
The drop comes from the **model-code refactor** (#212/#208/#226/#224 stripping norm-affine /
narrowing default MLPs), i.e. a symptom of the temporal drift, not a config typo.

### Cleared (not the cause)
k-Max-DeepLab #251 (`cross_attn_mode` defaults softmax, CLIC never sets kmeans); dynamic queries
#241 (`use_query_masks: false`); matcher refactor (parallelism only, scipy semantics unchanged);
`IncidenceBasedRegressionTask` energy logic functionally identical; `loss.py` only added a
`clamp_min(1.0)`; scaling yaml only whitespace. Known-intentional local adds not implicated.

### → DECISIVE NEXT EXPERIMENT (now task 5)
**Retrain from the `clic-paper` tag** (paper recipe: batch 512, devices 2), patching only the
data paths. If IQR reverts to falling → cause is the post-paper model refactor (candidates 1/2).
See `../05_reproduce_paper_tag/`.

---
## Original method notes (for reference)

## Objective
This repo's `origin` is **`lgray/hepattn`**, a fork of the original **`samvanstroud/hepattn`**
(the paper authors' repo). Find any local change — to the model, loss, matcher, task heads,
data reader, or the CLIC config — that drifted from the paper's code and could produce the
inverted IQR trend. A fork-side change is the **cleanest possible explanation** for the gap.

## Method
1. Add upstream as a remote and fetch (read-only):
   `git remote add upstream https://github.com/samvanstroud/hepattn.git && git fetch upstream`
2. Identify the merge-base / the commit the fork branched from, and the commit that produced
   the paper (check the paper / upstream tags/releases near arXiv 2508.20092, Aug 2025).
3. Diff **paths that affect the model, not the harness**, ranked by relevance:
   - `src/hepattn/experiments/clic/configs/base.yaml` ← **highest priority** (the 12M-vs-10.1M
     lead: dims, depth, heads, num_queries, head-MLP widths, incidence cutval, scaling dict).
   - `src/hepattn/models/` — `maskformer.py`, `decoder.py`, `encoder.py`, `attention.py`,
     `task.py`, `matcher.py`, `loss.py`.
   - `src/hepattn/experiments/clic/` — `predictionwriter.py`, `pflow_data.py`, readers.
   - `configs/clic_var_transform.yaml` (input/target scaling).
   `git diff upstream/<ref>...HEAD -- <path>` per group; summarize semantic (not cosmetic) diffs.
4. **Separate real drift from expected local additions:** this fork adds HPG SLURM scripts,
   the `studies/` work, doc files, and the local `attention.py` L4 fix + neutral-pt fix
   (both already analyzed in `../00_cross_run/`). Those are known/intentional — flag only
   changes to **model architecture, loss/matching, or CLIC config** that differ from the
   paper's recipe.

## Priority check (do first)
Reconcile **`base.yaml` params (10.1M) against the paper's stated 12M** at the upstream
paper-commit config. This is the single highest-value diff — it directly tests the leading lead.

## Checklist
- [ ] upstream remote added + fetched
- [ ] branch point / paper commit identified
- [ ] base.yaml diffed vs upstream (param reconciliation)
- [ ] models/ diffed
- [ ] clic experiment code diffed
- [ ] semantic drift summarized (vs. known-intentional local changes)

## Results
_(fill in)_
