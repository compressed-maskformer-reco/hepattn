"""Regenerate the CLIC eval ROOT file from the existing .h5 predictions, using the
fixed neutral-pt computation in predictionwriter.load_convert_h5.

Reads the existing .h5 (unchanged), writes a NEW .root with a distinct suffix so the
original epoch=195...__test.root is never overwritten. Replicates the ROOT-writing
block from PflowPredictionWriter.on_test_end verbatim.
"""

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic")
from predictionwriter import load_convert_h5  # noqa: E402

CKPT_DIR = Path(
    "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/"
    "logs/clic_v6_20260605-T113014/ckpts"
)
H5_PATH = CKPT_DIR / "epoch=195-val_loss=3.78800__test.h5"
# New, distinct output name — does NOT overwrite the original __test.root
ROOT_OUT = CKPT_DIR / "epoch=195-val_loss=3.78800__test_neutralptfix.root"

assert H5_PATH.exists(), f"missing input h5: {H5_PATH}"
assert not ROOT_OUT.exists(), f"refusing to overwrite existing output: {ROOT_OUT}"

print(f"Reading predictions from: {H5_PATH}")
event_number, pflow_class, pflow_ptetaphi, proxy_ptetaphi, pflow_indicator = load_convert_h5(H5_PATH.as_posix())

# Sanity: how many neutral pt values actually changed vs the old no-op behavior
neutral_mask = (pflow_class < 5) & (pflow_class > 2)
print(f"neutral objects: {int(neutral_mask.sum())}")
print(f"neutral pt after fix: mean={pflow_ptetaphi[neutral_mask, 0].mean():.3f} "
      f"max={pflow_ptetaphi[neutral_mask, 0].max():.3f}")

print(f"Writing ROOT file to: {ROOT_OUT}")
with uproot.recreate(ROOT_OUT.as_posix()) as f:
    f["event_tree"] = {
        "mpflow": {
            "pt": ak.Array(pflow_ptetaphi[..., 0]),
            "eta": ak.Array(pflow_ptetaphi[..., 1]),
            "phi": ak.Array(pflow_ptetaphi[..., 2]),
            "class": ak.Array(pflow_class),
        },
        "proxy": {
            "pt": ak.Array(proxy_ptetaphi[..., 0]),
            "eta": ak.Array(proxy_ptetaphi[..., 1]),
            "phi": ak.Array(proxy_ptetaphi[..., 2]),
        },
        "pred_ind": ak.Array(pflow_indicator),
        "event_number": ak.Array(event_number)[: len(pflow_indicator)],
    }
print(f"DONE. Wrote {ROOT_OUT} ({ROOT_OUT.stat().st_size/1e6:.1f} MB)")
