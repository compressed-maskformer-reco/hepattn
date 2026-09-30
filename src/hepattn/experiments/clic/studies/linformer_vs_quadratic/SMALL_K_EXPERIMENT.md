# Small-k encoder-only Linformer — experiment brief

**Audience:** Helen, and the Claude session helping her run it on ALCF Polaris.
**Written:** 2026-09-10, by Erdem's Claude session, from the `linformer-float` branch of
`erdemyigit/hepattn`.
**Everything below that states a number was measured, not estimated.** Where something is
an assumption, it says so.

---

## 1. What you are running, in one paragraph

Train the CLIC particle-flow MaskFormer with **Linformer attention in the encoder only, at
projection rank k=32**, and quadratic attention in the decoder. Two runs: one on the hits in
file order, one with the hits sorted in phi. Each is a 200-epoch run on one Polaris node
(4x A100-40GB). Everything else is held fixed against the arms already running.

---

## 2. Why this experiment matters

### 2.1 No run so far actually compresses anything

Linformer replaces the n-by-n attention matrix with a learned projection `E` of shape
`(n, k)`, so cost scales as `n*k` instead of `n^2`. That is only a saving when `k < n`.

In CLIC the encoder sequence is **n = 168**: 160 `max_nodes` plus 8 register tokens. Every
Linformer run to date has used the library default **k = 256**. Since 256 > 168, the
projection *expands* the sequence, saves nothing, and adds parameters. The efficiency claim
the whole study rests on has never been tested.

Measured parameter counts, built from the actual YAML configs:

| arm | total params | encoder params |
|---|---|---|
| Linformer k=256 (all runs so far) | 12,451,027 | 3,953,152 |
| **k=32 encoder-only (this experiment)** | **10,165,459** | **3,231,232** |
| quadratic control (Erdem is running this) | 10,126,115 | 3,191,888 |

The k=32 arm sits **within 0.4% of the quadratic control**. That makes it a
parameter-matched comparison. The k=256 arm never was: it carried 2.3M extra parameters,
which confounds any claim that Linformer is cheaper.

Break-even against quadratic is `k = n/2 = 84`, so k=32 is a genuine 5.25x compression of
the sequence axis and comfortably inside the win region.

### 2.2 Why the encoder only

The decoder cannot take small k. Its cross-attention uses a **per-query** attention mask of
shape `(batch, num_queries, num_hits)`, and Linformer projects K and V **once**, shared
across all queries. A per-query key mask cannot be expressed after that shared projection.
Setting k < n in the decoder raises a `RuntimeError` on the mask expand. This is structural,
not a bug to patch.

The encoder never passes a per-query mask. Its only mask is padding, which the code handles
by zeroing rows *before* the sequence projection. So the encoder has no k constraint, and
**no code change is needed** to run small k there.

### 2.3 Why phi sorting, and why it belongs with small k

`E` is indexed by **absolute sequence position**. Column j of the compressed sequence is a
fixed weighted sum over positions. The CLIC reader concatenates `[tracks, topoclusters]` in
production order, so position i means nothing from one event to the next, and `E` can only
learn a global average.

Quadratic attention does not care, because it is permutation-equivariant, and the positional
encoder is content-based: it reads each node's own eta and phi, never its index. Linformer
is the one component that is position-sensitive.

Sorting the hits gives position a geometric meaning so the projection can learn a spatial
pooling. Prior art: the only published Linformer-on-FPGA result on this stack
(Sun et al., arXiv:2510.24784, section 2.1) sorts constituents by pT and runs k=2
successfully. They do not discuss the ordering-projection interaction, but they do sort.

**The key point:** at k=256 the projection is not summarizing anything, so a sorting test
there can come back null for an uninteresting reason. At k=32 each of the 32 columns must
pool roughly five positions, and whether those pools are geometrically coherent is exactly
what sorting decides. Your two runs are where the sorting hypothesis actually gets tested.

### 2.4 The design you are completing

| | k=256, no compression | k=32, 5.25x compression |
|---|---|---|
| **hits in file order** | Erdem, epoch 199, done, val/loss 4.51461 | **your run A** |
| **hits sorted in phi** | Akum, running now | **your run B** |

If you can only get one job through, **run B** (k=32 + phi sort). It pairs against Akum's
run as a single change in k, and it is the configuration you would actually deploy.

---

## 3. Prerequisites

1. An ALCF Polaris account with an allocation you can charge (`-A` value).
2. The CLIC ROOT files staged somewhere on `/eagle`. You need
   `train_clic_fix.root` and `val_clic_fix.root`. Roughly 12 GB. Ask Erdem or Akum for the
   path they use rather than re-downloading.
3. A Python venv built from **this tree's own lock** (`uv python install 3.12.0 && uv sync
   --no-install-project --python 3.12.0` in the clone), with `VENV_DIR` in `env.sh` pointed at
   it. Do not reuse someone else's venv: it is only executable by its owner, and a venv built
   from a different lock can carry a lightning whose logger arguments the configs do not match.
4. This repo checked out on Polaris, on the branch described in section 4.

**You do not need Erdem's checkpoints.** These are fresh runs from scratch.

---

## 4. Step 0 — get the code

The experiment needs three things that live on the `linformer-float` branch of
`erdemyigit/hepattn`:

- the padding-leak fix (`7c36e32`), which stops padded hits contaminating real tokens
  through the sequence projection;
- the qkv-norm fix (`bb2b1ca`), which applies the q/k/v norms on the Linformer path where
  they previously received no gradient at all;
- the `sort_nodes_by` data option and the two configs, described in appendices A and B.

```bash
git clone https://github.com/erdemyigit/hepattn.git
cd hepattn
git checkout linformer-float
```

**Check whether the pieces are already there:**

```bash
grep -q "sort_nodes_by" src/hepattn/experiments/clic/pflow_data.py && echo "sort: present" || echo "sort: MISSING, apply Appendix B"
ls src/hepattn/experiments/clic/configs/link32_sortphi_polaris.yaml 2>/dev/null && echo "configs: present" || echo "configs: MISSING, apply Appendix A"
```

Both are committed on `linformer-float`. If the checks above say MISSING you are on a stale
checkout: `git pull` on a login node. The appendices reproduce them exactly as a fallback.

**Do not use the built-in `hepattn.utils.sorter.Sorter` for CLIC.** It looks like the
three-line option, but it sorts the mask target and the node features while leaving the
incidence truth (`particle_incidence`) and the raw inputs the regression proxy reads
untouched. Measured on a real event: after sorting, the mask target and the incidence
target disagree on 5 of 30 cells. The data-reader option below moves everything together.

---

## 5. Step 1 — create `polaris/env.sh`

This file is **gitignored on purpose**, because it holds site-specific values. Every submit
script sources it. Create it yourself:

```bash
cat > polaris/env.sh <<'EOF'
export PBS_PROJECT="CHANGEME"           # your allocation, the -A value
export PBS_FILESYSTEMS="home:eagle"     # a job that omits a filesystem it touches gets held
export QUEUE_DEBUG="debug"              # 1-2 nodes, 1h. Smoke tests.
export QUEUE_PROD="preemptable"         # 1-10 nodes, 72h, can be killed at any time
export DATA_ROOT="/eagle/CHANGEME/clic" # where train_clic_fix.root and val_clic_fix.root live
export WORK_ROOT="/eagle/CHANGEME/hepattn-helen"
export WALLTIME="20:00:00"              # see the queue note in section 12
export PYTHONNOUSERSITE=1               # never let ~/.local shadow the venv
EOF
```

If you are reusing someone else's venv, also set `VENV_DIR` to it. Otherwise the PBS script
defaults to `${WORK_ROOT}/hepattn/.venv` and will fail loudly if nothing is there.

---

## 6. Step 2 — verify the configs

The two configs differ from the already-validated `linformer_polaris.yaml` in a *very* small
number of lines. Confirm that before submitting anything:

```bash
cd src/hepattn/experiments/clic/configs
diff linformer_polaris.yaml link32_polaris.yaml
diff link32_polaris.yaml link32_sortphi_polaris.yaml
```

Expected, and nothing else:

- `name:` changes.
- Encoder gains `linformer_proj_dim: 32` and `linformer_seq_len: 168`.
- Decoder loses `attn_type: linformer`, so it falls back to the default `torch`.
- The sorted twin adds `sort_nodes_by: phi` under `data:`.

If you see anything beyond that, stop and ask. A controlled comparison is the entire point.

---

## 7. Step 3 — local smoke test, no GPU needed

**On a Polaris login node** prefix the command with `CC=gcc CXX=g++`: inductor otherwise picks
NVHPC's `nvc++` and fails with `nvc++-Error-Unknown switch: -fno-trapping-math`. The PBS scripts
already export this; the login-node shell does not.

Run this **before** touching the queue. It builds each arm from its YAML, does a real
forward and backward on dummy data, and proves the projection is the right shape and gets
gradient. Takes under a minute on a laptop.

```bash
cd src/hepattn/experiments/clic
PYTHONPATH=../../.. python - <<'EOF'
import yaml, warnings, torch
warnings.filterwarnings("ignore")
from jsonargparse import ArgumentParser
from hepattn.models.maskformer import MaskFormer
from hepattn.experiments.clic.pflow_data import CLICDataset
from torch.utils.data import DataLoader

p = ArgumentParser(); p.add_subclass_arguments(MaskFormer, "model")
cfg = yaml.safe_load(open("configs/link32_sortphi_polaris.yaml"))
model = p.instantiate_classes(p.parse_object({"model": cfg["model"]["model"]})).model
print("total params:", sum(q.numel() for q in model.parameters()))

ds = CLICDataset(filepath="", inputs=cfg["data"]["inputs"], targets=cfg["data"]["targets"],
                 scale_dict_path="configs/clic_var_transform.yaml", dummy_data=True,
                 num_events=4, max_nodes=160, num_objects=150)
inputs, targets = next(iter(DataLoader(ds, batch_size=2)))
out = model(inputs)
_, _, losses = model.loss(out, targets)
tot = sum(v for layer in losses.values() for t in layer.values() for v in t.values())
tot.backward()
pk = model.encoder.layers[0].attn.fn.attn.proj_k
print("proj_k shape:", tuple(pk.shape), "grad flows:", bool(pk.grad.abs().sum() > 0))
EOF
```

**Expected output:**

```
total params: 10165459
proj_k shape: (168, 32) grad flows: True
```

`proj_k` being `(168, 32)` is the whole experiment. If it comes back `(256, 256)` your
`attn_kwargs` did not reach the attention module and the run would silently be the old
uncompressed arm.

**One thing that looks alarming and is not:** if you print the individual loss terms, the
`mask_bce` terms come back `inf` on dummy data. That happens identically on the k=256 arm
and the quadratic arm, because the dummy targets contain rows with no valid constituents.
It is an artifact of the fake data, not of this config. It does not occur on real data.

---

## 8. Step 4 — Polaris smoke test

Thirty minutes on the debug queue, on real GPUs and real data, before you commit to a long
run:

```bash
SMOKE=1 CFG=src/hepattn/experiments/clic/configs/link32_sortphi_polaris.yaml \
  bash polaris/submit_linformer.sh
```

This runs 100 train batches and 10 val batches for one epoch. Watch
`polaris/logs/clic-link32_sortphi-smoke.log`.

**What good looks like:** a `preflight OK` line, four A100s listed, a finite decreasing loss,
and an it/s figure you can use to extrapolate the full wall time.

The smoke path deliberately uses at least 40 steps. Fewer, and the OneCycleLR schedule
divides by zero during phase 1.

---

## 9. Step 5 — the full runs

```bash
# Run B, the priority one: k=32 with phi sorting
CFG=src/hepattn/experiments/clic/configs/link32_sortphi_polaris.yaml \
  bash polaris/submit_linformer.sh

# Run A, the unsorted control
CFG=src/hepattn/experiments/clic/configs/link32_polaris.yaml \
  bash polaris/submit_linformer.sh
```

Both train 200 epochs on 4 GPUs with DDP, batch size 256, AdamW, OneCycleLR. Logs land in
`polaris/logs/clic-link32*.log`; checkpoints under
`src/hepattn/experiments/clic/logs/<run name>-<timestamp>/ckpts/`.

The `preemptable` queue can kill your job at any time. The config checkpoints every epoch,
so resume with:

```bash
RESUME_CKPT=/abs/path/to/last.ckpt CFG=<same config> bash polaris/submit_linformer.sh
```

---

## 10. Step 6 — evaluate

```bash
CKPT=/abs/path/to/epoch=199-val_loss=X.XXXXX.ckpt bash polaris/submit_linformer_eval.sh
```

This writes `<ckpt name>__test.h5` and `__test.root` next to the checkpoint. Send Erdem the
`.root` file; he has the jet-clustering and performance plotting pipeline set up locally.

**Two things to tell him when you send it,** because they have already caused confusion once:

- Whether you evaluated on `test_clic_common_infer.root` or `val_clic_fix.root`. Comparisons
  are only valid on a common file.
- That the prediction branches are named with a dot, `mpflow.pt`, on this branch. Akum's
  tree uses `mpflow_pt` with an underscore. Erdem's reader auto-detects, but say which.

---

## 11. What to report back

The primary number is **jet energy-response IQR versus jet energy**, but the cheap first
signal is validation loss. For reference, the k=256 unsorted arm finished at
**val/loss 4.51461** at epoch 199, best 4.51358 at epoch 191.

Report, for each of your two runs:

| quantity | why |
|---|---|
| final and best `val/loss`, with epoch | direct comparison against 4.51461 |
| wall-clock time per epoch | the efficiency claim needs a measured number, not a FLOP count |
| peak GPU memory | second axis of the efficiency claim |
| the `__test.root` file | so Erdem can produce the physics plots |

The two questions your runs answer:

1. **Does k=32 cost accuracy?** Compare run A against the k=256 unsorted arm at 4.51461. If
   the loss is comparable, an 5.25x sequence compression is free, and the study finally has
   an efficiency result.
2. **Does sorting help, where it should?** Compare run B against run A. This is the clean
   test of Lindsey's hypothesis, at the only k where the projection has real work to do.

A null result on question 2 is still a real result, and worth reporting plainly.

---

## 12. Known failure modes

Every one of these has actually happened on this project. They are ordered by how much time
they cost.

**Two checkouts, wrong code trained.** `hepattn` is deliberately *not* installed into the
venv, so `PYTHONPATH` alone decides which clone runs. If there is more than one checkout, a
job can train the wrong branch and look perfectly healthy. The PBS script has a preflight
that asserts `hepattn` resolves inside `$REPO_DIR`. Do not remove it.

**`nvc` instead of `gcc`.** Polaris puts NVHPC's `nvc` on PATH as the default C compiler.
Triton builds its CUDA helper with `$CC` and passes GCC-only flags, which `nvc` rejects with
`nvc-Error-Unknown switch: -Wno-psabi`. The PBS script forces `CC=gcc`. This is not avoidable
by dropping torch.compile, because the loss module wraps cost functions at import time.

**Long queue wait on `preemptable`.** A 48-hour walltime request is hard to backfill. Erdem's
quadratic job sat in `Q` for a long time for exactly this reason. Extrapolate a realistic
wall time from the smoke test's it/s and request that plus margin, via `WALLTIME` in
`env.sh`. Diagnose a stuck job with `qstat -f <jobid> | grep -i comment`.

**`git pull` fails on a compute node.** Compute nodes have no outbound network. Pull on a
login node before submitting.

**`torch.compile` on the Linformer backward.** Inductor miscompiles it. The `Compile`
callback is already removed from these configs. Do not add it back for a Linformer arm.

**Stale PBS script.** `submit_linformer.sh` resolves `REPO_DIR` from its own location. If you
have several clones, make sure you are invoking the script from the tree you actually
updated.

---

## Appendix A — regenerate the two configs

Run from the repo root. It derives both configs from `linformer_polaris.yaml` and asserts on
every edit, so it fails loudly rather than silently producing something different.

```bash
cd src/hepattn/experiments/clic/configs
python3 - <<'EOF'
from pathlib import Path
src = Path("linformer_polaris.yaml").read_text()

old_enc = """          attn_type: linformer
          hybrid_norm: true
          value_residual: false
          num_register_tokens: 8
          attn_kwargs:
            num_heads: 16"""
new_enc = """          attn_type: linformer
          hybrid_norm: true
          value_residual: false
          num_register_tokens: 8
          attn_kwargs:
            num_heads: 16
            # n = 160 max_nodes + 8 register tokens. k=32 is a real 5.25x compression
            # of the sequence axis; break-even vs quadratic is k = n/2 = 84.
            linformer_proj_dim: 32
            linformer_seq_len: 168"""
assert src.count(old_enc) == 1
s = src.replace(old_enc, new_enc)

# Decoder returns to quadratic: its per-query attn_mask cannot survive a shared K/V
# projection, which is what forces k >= n there.
old_dec = """          attn_kwargs:
            num_heads: 16
            attn_type: linformer"""
new_dec = """          attn_kwargs:
            num_heads: 16"""
assert s.count(old_dec) == 1
s = s.replace(old_dec, new_dec)

base = s.replace("name: clic_v6_linformer", "name: clic_v6_link32")
Path("link32_polaris.yaml").write_text(base)

sorted_cfg = base.replace("name: clic_v6_link32", "name: clic_v6_link32_sortphi").replace(
    "  incidence_cutval: 0.01",
    "  incidence_cutval: 0.01\n  # Give the position-indexed projection a geometric meaning.\n  sort_nodes_by: phi")
Path("link32_sortphi_polaris.yaml").write_text(sorted_cfg)
print("wrote link32_polaris.yaml and link32_sortphi_polaris.yaml")
EOF
```

---

## Appendix B — the `sort_nodes_by` data option

Only needed if the grep in section 4 said MISSING. Run from the repo root. Same
assert-on-every-edit approach.

The change adds four things to `src/hepattn/experiments/clic/pflow_data.py`: a `sort_nodes_by`
constructor argument with validation, a module-level `node_sort_order` helper, application of
the permutation to every per-node tensor, and application of **the same** permutation to the
incidence-matrix columns.

That last part is the one that matters. The node features and the truth incidence must move
together or the truth is silently misaligned with the inputs.

```bash
python3 - <<'PYEOF'
from pathlib import Path
p = Path("src/hepattn/experiments/clic/pflow_data.py")
s = p.read_text()
if "sort_nodes_by" in s:
    raise SystemExit("already present, nothing to do")

def rep(old, new):
    global s
    assert s.count(old) == 1, f"expected exactly one match for:\n{old}"
    s = s.replace(old, new)

rep("""        is_inference: bool = False,
        dummy_data: bool = False,
    ):
        super().__init__()
""", """        is_inference: bool = False,
        dummy_data: bool = False,
        sort_nodes_by: str | None = None,
    ):
        super().__init__()
""")

rep("""        self.is_inference = is_inference

        if dummy_data:""", """        self.is_inference = is_inference
        if sort_nodes_by is not None and sort_nodes_by not in NODE_SORT_MODES:
            raise ValueError(f"sort_nodes_by must be one of {sorted(NODE_SORT_MODES)} or None, got {sort_nodes_by!r}")
        self.sort_nodes_by = sort_nodes_by

        if dummy_data:""")

rep('''def do_padding(tensor, max_len):
    x = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
    x[: len(tensor)] = tensor
    return x
''', '''def do_padding(tensor, max_len):
    x = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
    x[: len(tensor)] = tensor
    return x


# Orderings for the node (track + topocluster) sequence. The default file order is
# [tracks, topos], each in production order, so sequence position i has no stable meaning
# across events. Quadratic attention does not care (it is permutation-equivariant; the
# positional encoder acts on each node's own eta/phi, not its index), but Linformer's
# projection E[n, k] is indexed by position, so it can only learn a spatial pooling if
# position correlates with geometry. Sorting gives it that correlation.
NODE_SORT_MODES = frozenset({"phi", "eta", "type_phi", "type_eta"})


def node_sort_order(mode: str, is_track: torch.Tensor, phi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Return the permutation that sorts the unpadded node sequence.

    ``phi``/``eta`` sort all nodes together by that coordinate. ``type_phi``/``type_eta``
    keep tracks before topoclusters and sort within each block. Sorts are stable, so
    ties keep file order.
    """
    key = {"phi": phi, "eta": eta, "type_phi": phi, "type_eta": eta}[mode]
    order = torch.argsort(key.double(), stable=True)
    if mode.startswith("type_"):
        # second stable sort on the block key: tracks (1) first, topos (0) after
        block = (1 - is_track.double())[order]
        order = order[torch.argsort(block, stable=True)]
    return order
''')

rep('''        for key, val in node_features.items():
            val = do_padding(val, self.max_nodes)
            node_features[key] = val

        for key, val in node_raw_features.items():
            val = do_padding(val, self.max_nodes)
            node_raw_features[key] = val
''', '''        # Permute every per-node tensor with one shared order; the incidence columns are
        # permuted with the same order below, after the matrix is built in file order.
        order = None
        if self.sort_nodes_by is not None:
            order = node_sort_order(
                self.sort_nodes_by,
                node_raw_features["is_track"],
                node_raw_features["raw_phi"],
                node_raw_features["raw_eta"],
            )

        for key, val in node_features.items():
            if order is not None:
                val = val[order]
            val = do_padding(val, self.max_nodes)
            node_features[key] = val

        for key, val in node_raw_features.items():
            if order is not None:
                val = val[order]
            val = do_padding(val, self.max_nodes)
            node_raw_features[key] = val
''')

rep('''        incidence = torch.tensor(incidence_matrix, dtype=torch.float32)

        incidence = torch.nn.functional.pad(incidence, (0, self.max_nodes - n_nodes, 0, 0))
''', '''        incidence = torch.tensor(incidence_matrix, dtype=torch.float32)
        if order is not None:
            incidence = incidence[:, order]

        incidence = torch.nn.functional.pad(incidence, (0, self.max_nodes - n_nodes, 0, 0))
''')

p.write_text(s)
print("patched", p)
PYEOF
```

There is a test file for this at `tests/experiments/clic/test_node_sort.py` on Erdem's tree.
It has 15 tests. Each was verified to be discriminating by mutation: breaking the incidence
permutation fails 4 tests, breaking the feature permutation fails 4, replacing the sort with
identity fails 8, and dropping the track/topo block key fails 1. Ask for it if you want to
re-verify the patch applied correctly.
