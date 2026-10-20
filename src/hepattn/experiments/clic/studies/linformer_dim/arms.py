"""One table of trainings, shared by every plot script in this study (after the paper-tag studies/linformer/arms.py).

The decoder-only Linformer k scan: the linformer_size small reference on one B200 at batch 2048, with the decoder's
linformer_k halved from 64 to 4 and the encoder left at 64. Two training settings, each a full scan:
  scan        peak LR 3.2e-4, weight_decay 1e-4, 200 epochs
  scan_wd0p1  peak LR 6.2e-4, weight_decay 0.1, 400 epochs
and `settings`, the two settings' k=64 and k=4 arms side by side.

WHICH RUNS ARE DRAWN is chosen by the ARM_SET environment variable (default `scan`), and every script honours it.
Each set has its own figure and cache prefix, so drawing one never overwrites another. In every set the first arm
is the baseline every delta is taken against.

Every arm is drawn at its lowest-val_loss checkpoint, the one submit_eval_l4.sh resolves for a TRAIN_JOB evaluation.
The stems below were read from the runs, and the parameter counts from their metadata.yaml.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# The runs were trained from mmazza's clone, whose clic/logs links here; named directly so other clones read them.
LOGS = Path("/cmsuf/data/store/user/mmazza/fastml/compressed-maskformer-reco/hepattn/src/hepattn/experiments/clic/logs")


@dataclass(frozen=True)
class Arm:
    """One trained run, and everything the plot scripts need to find and draw it."""

    key: str  # short id; also the network name inside the Performance pipeline
    label: str  # what a legend shows
    k: int  # decoder linformer_k
    folder: str  # run directory under logs/
    stem: str  # checkpoint stem, without the .ckpt / __test.root suffix
    params: int  # trainable_params from the run's metadata.yaml
    color: str
    photon: str  # lightened companion hue, for the per-class figure's photon curves
    marker: str
    dash: object
    heavy: bool = False  # drawn at reference weight

    @property
    def root(self) -> Path:
        """The evaluation output the physics figures read."""
        return LOGS / self.folder / "ckpts" / f"{self.stem}__test.root"

    @property
    def metrics(self) -> Path:
        """The CSVLogger file the training-curve figures read."""
        return LOGS / self.folder / "csv_metrics" / "metrics.csv"


SOLID = "-"
DASH = (0, (5, 2))
DOT = (0, (1, 1.2))
DASHDOT = (0, (4, 1.5, 1, 1.5))
DASHDOTDOT = (0, (3, 1.5, 1, 1.5))

# fmt: off
_SCAN = [
    Arm("deck64", "dec k=64 (ref) 1.19M", 64, "clic_small_linformer_deck64_20261013-T180136", "epoch=197-val_loss=4.38151", 1_194_467, "#003f5c", "#4a90b8", "o", SOLID, heavy=True),  # noqa: E501
    Arm("deck32", "dec k=32 1.07M", 32, "clic_small_linformer_deck32_20261013-T180142", "epoch=195-val_loss=4.48544", 1_071_587, "#58508d", "#9b93c4", "s", DASH),  # noqa: E501
    Arm("deck16", "dec k=16 1.01M", 16, "clic_small_linformer_deck16_20261013-T180138", "epoch=198-val_loss=4.50879", 1_010_147, "#bc5090", "#ff9db0", "D", DOT),  # noqa: E501
    Arm("deck8", "dec k=8 979k", 8, "clic_small_linformer_deck8_20261013-T180136", "epoch=197-val_loss=4.54631", 979_427, "#ff6361", "#ffb0ae", "^", DASHDOT),  # noqa: E501
    Arm("deck4", "dec k=4 964k", 4, "clic_small_linformer_deck4_20261013-T180447", "epoch=198-val_loss=4.48444", 964_067, "#ffa600", "#ffd280", "v", DASHDOTDOT),  # noqa: E501
]
_WD = "clic_small_linformer_lr6p2e-4_wd0p1_ep400"
_SCAN_WD = [
    Arm("wd_deck64", "dec k=64 (ref) 1.19M", 64, f"{_WD}_deck64_20261015-T153629", "epoch=399-val_loss=4.17905", 1_194_467, "#003f5c", "#4a90b8", "o", SOLID, heavy=True),  # noqa: E501
    Arm("wd_deck32", "dec k=32 1.07M", 32, f"{_WD}_deck32_20261015-T160935", "epoch=394-val_loss=4.19195", 1_071_587, "#58508d", "#9b93c4", "s", DASH),  # noqa: E501
    Arm("wd_deck16", "dec k=16 1.01M", 16, f"{_WD}_deck16_20261015-T172925", "epoch=394-val_loss=4.22542", 1_010_147, "#bc5090", "#ff9db0", "D", DOT),
    Arm("wd_deck8", "dec k=8 979k", 8, f"{_WD}_deck8_20261015-T181235", "epoch=393-val_loss=4.18977", 979_427, "#ff6361", "#ffb0ae", "^", DASHDOT),
    Arm("wd_deck4", "dec k=4 964k", 4, f"{_WD}_deck4_20261015-T181243", "epoch=398-val_loss=4.29525", 964_067, "#ffa600", "#ffd280", "v", DASHDOTDOT),
]
# fmt: on


def _relabel(arm, suffix, dash, color=None, photon=None, marker=None):
    """The same run under a label that names its training setting, for a set that mixes the two.

    The setting is told apart by colour and marker as well as dash, because the jet-IQR figures draw every line solid.
    """
    over = {"label": f"dec k={arm.k} {suffix}", "dash": dash}
    over |= {key: val for key, val in (("color", color), ("photon", photon), ("marker", marker)) if val is not None}
    return Arm(**{**arm.__dict__, **over})


_SETTINGS = [
    _relabel(_SCAN[0], "lr3.2e-4 wd1e-4 200ep", SOLID),
    _relabel(_SCAN_WD[0], "lr6.2e-4 wd0.1 400ep", DASH, color="#2f9bd6", photon="#9fd0ef", marker="s"),
    _relabel(_SCAN[4], "lr3.2e-4 wd1e-4 200ep", SOLID),
    _relabel(_SCAN_WD[4], "lr6.2e-4 wd0.1 400ep", DASH, color="#a35a00", photon="#d9a36b", marker="^"),
]
_COMMON = "1 B200 at batch 2048, GPU matcher; evaluated at the lowest-val_loss checkpoint with the host solver"

# name -> (figure and cache prefix, title, arms)
SETS = {
    "scan": (
        "linformer_dim",
        "Decoder-only Linformer k scan on the small linformer model — encoder k=64 throughout\n"
        f"peak LR 3.2e-4, weight_decay 1e-4, 200 epochs, {_COMMON}",
        _SCAN,
    ),
    "scan_wd0p1": (
        "linformer_dim_wd0p1",
        "Decoder-only Linformer k scan on the small linformer model — encoder k=64 throughout\n"
        f"peak LR 6.2e-4, weight_decay 0.1, 400 epochs, {_COMMON}",
        _SCAN_WD,
    ),
    "settings": (
        "linformer_dim_settings",
        "The two training settings at decoder k=64 and k=4 — encoder k=64 throughout\n"
        f"solid: peak LR 3.2e-4, weight_decay 1e-4, 200 epochs   |   dashed: peak LR 6.2e-4, weight_decay 0.1, 400 epochs   |   {_COMMON}",
        _SETTINGS,
    ),
}

ARM_SET = os.environ.get("ARM_SET", "scan")
if ARM_SET not in SETS:
    raise SystemExit(f"ARM_SET={ARM_SET!r} unknown; pick one of {sorted(SETS)}")
PREFIX, TITLE, ARMS = SETS[ARM_SET]
BASELINE = ARMS[0]
print(f"=== ARM_SET={ARM_SET}  baseline={BASELINE.label}  arms={[a.label for a in ARMS[1:]]}", flush=True)

# Pandora is the classical reference the performance notebook always plots; filled and receding, not an arm.
PANDORA_COLORS = {"neut had": "#999999", "photon": "#cccccc"}
