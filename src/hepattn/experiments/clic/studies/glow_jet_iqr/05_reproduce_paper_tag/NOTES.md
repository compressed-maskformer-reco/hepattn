# Task 5 — Reproduce from the `clic-paper` tag (THE decisive experiment)

**Status:** ⬜ TODO — created 2026-07-07 as a direct consequence of task 4.

## Why this is now the priority
Task 4 proved we are **not training the paper's model**: our `HEAD` is 100 commits + a model
refactor past the paper's `clic-paper` tag (`fb90390`, 2025-08-11). The rising-vs-falling jet-E
IQR is most likely caused by that refactor (leading suspect: incidence-head width [256,256]→
[512,512]; plus norm/decoder-attention rewrite). The one experiment that cleanly settles it:
**train the actual paper code and see if the IQR trend reverts to falling.**

## Setup ACTUALLY DONE (2026-07-07) — separate clean clone (per user: independent, off the fork)
- **Location:** `/home/m.mazza/blue/projects/fastml/hepattn-clic-paper`
  (= `/blue/avery/m.mazza/projects/fastml/hepattn-clic-paper`), a **fresh clone of the ORIGINAL
  `samvanstroud/hepattn`** (NOT the `lgray` fork), checked out at tag **`clic-paper` (fb90390)**.
  Kept fully separate so it can't disturb the fork workflow.
- **Env:** built the paper tag's OWN pixi env from its `pixi.lock` via `pixi install` (torch 2.7.0
  cu126, py3.12, flash-attn 2.7.4 prebuilt wheel). Runs via **direct `pixi run`** (no apptainer),
  so model code + deps match the paper snapshot exactly. `pixi` = `/home/m.mazza/.pixi/bin/pixi`.
- **Config:** patched only `src/.../clic/configs/base.yaml` data paths → our
  `/blue/avery/.../data/clic/{train_clic_fix,val_clic_fix,test_clic_common_infer}.root`
  (same dataset/filenames as the paper's `/share/gpu1/...`), and `batch_size 512 → 170`
  (170×6 L4 = global 1020 ≈ paper's 512×2=1024; L4 can't hold 512/GPU). Everything else is the
  pristine paper recipe (dim 256, enc6/dec4, flash-varlen, bf16-mixed, Lion @8e-5, 200 epochs).
- **Submit:** `src/.../clic/submit_training_hpg_l4_2nodes.sh` — 6×L4 (2 nodes), `COMET_MODE=offline`
  (paper tag uses plain CometLogger; force offline so it doesn't try the internet), `srun pixi run
  python main.py fit ... --trainer.devices=3 --trainer.num_nodes=2`.
- **Param-count control:** `ModelSummary` callback prints total params at startup → verify ≈**12M**
  in the slurm log (positive control that we're running the paper's larger model).

## SUBMITTED 2026-07-07 → job 36526156 (PENDING), 6×L4 (2 nodes)
Env build was non-trivial (documented so it's reproducible):
1. Paper `pixi.lock` was stale for the republished `cuda-version-12.9-3.conda` metapackage
   (hash mismatch). Fixed by `pixi lock` (backup `pixi.lock.orig`). Verified the relock left the
   model stack UNCHANGED: torch 2.7.0/cu126, flash-attn 2.7.4 wheel, python 3.12.11, numpy,
   lightning 2.5.0, numba, awkward, uproot all identical; only peripheral deps + scipy 1.16.0→1.16.2
   (Hungarian matcher output unaffected) drifted.
2. pixi 0.70.1 passes a `pixi-conda-environment` config-setting that hepattn's build backend
   (scikit-build-core) rejects → editable build failed → pixi's all-or-nothing PyPI phase installed
   NOTHING. Workaround: commented out `hepattn = {path=".",editable=true}` in pyproject
   `[tool.pixi.pypi-dependencies]` (backup `pyproject.toml.orig`) so pixi installs torch/flash-attn/
   etc., then built hepattn with **plain pip**: `pixi run python -m pip install -e .
   --ignore-requires-python` (build isolation ON; `--ignore-requires-python` bypasses the paper's
   strict `requires-python == "3.12"` which plain pip reads as 3.12.0). hepattn's compiled part is a
   pure pybind11 `_core` module (no CUDA/torch). Confirmed hepattn survives `pixi install` reconcile.
3. Submit script uses `srun pixi run --frozen` (no network / no lock race across the 6 tasks on
   offline compute nodes). pixi = /home/m.mazza/.pixi/bin/pixi.
- Config confirmed: our data paths, batch 170 (global ~1020), bf16-mixed, flash-varlen, dim 256 —
  the paper recipe (NOT fp32/torch; that's task 3). ModelSummary will print param count at start
  → expect **~12M** (positive control we're running the paper's larger model).
- Output folder: `hepattn-clic-paper/.../logs/clic_v6_<timestamp>` (paper config `name: clic_v6`).

## After training
Eval with the SAME pipeline and overlay IQR vs the HEAD runs + paper Fig. 4. Interpretation below.

## Interpretation
- IQR reverts to **falling** with energy → confirmed: the discrepancy is the post-paper refactor.
  Next: bisect which change (incidence width vs norm/decoder) by toggling on `clic-paper`.
- IQR still **rises** → the model code is not the cause; look upstream (data/targets/recipe),
  though those were largely cleared in Phase 1.

## Open questions / watch-outs
- The paper-tag code may need an older env than the current `pixi.sif` (100 commits of dep drift).
  Check whether the paper-tag repo builds/runs in our container; may need a matching env.
- `flash-varlen` attention on the paper tag — same GPU/precision caveats as HEAD.
- Decide compute: this likely wants the same 6×L4 (2 nodes) slot — may compete with the fp32 job
  (36518920) and the running v7 jobs for nodes.

## Checklist
- [ ] worktree on `clic-paper` created
- [ ] data paths patched; config/env confirmed to run in container
- [ ] param count verified ≈12M
- [ ] training submitted → 200 epochs
- [ ] eval'd → IQR overlay vs HEAD runs
- [ ] verdict (trend reverts? Y/N) + which change bisected

## Results
_(fill in)_
