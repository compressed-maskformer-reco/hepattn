"""Jet-E response median and IQR per truth-E bin for every decoder-k arm, against the k=64 reference.

Ported from the paper-tag studies/linformer/plot_linformer_jet_iqr.py. Reconstructed particles are clustered into
jets and matched to truth jets on kinematics (how Pandora is scored), so the reading does not depend on the loss
matching. Every delta carries its bootstrap sigma_stat only: it measures the finiteness of the test sample, not the
spread between two trainings of one config, which is not measured for this model. Read the output as an ordering.

The default is the `mpflow_proxy` convention; CONVENTIONS=mpflow,mpflow_proxy adds the regression-head one and the
sign check (a real difference moves both conventions the same way).

Run from the clic experiment dir (first time via submit_plots.sh; jet clustering wants cores):
    pixi run -e clic python studies/linformer_dim/plot_jet_iqr.py
"""

import hashlib
import os
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from hepattn.experiments.clic.performance.performance import NetworkType, Performance, PerformanceConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ARMS, BASELINE, PREFIX, TITLE

PANDORA = NetworkType.PANDORA.value
PANDORA_STYLE = {"color": "#888888", "lw": 2.0, "ls": "-", "zorder": 1.5}

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CACHE_DIR = OUT / "cache"
# The truth-level test sample, in the data directory the submit scripts read ($CLIC_DATA, same default).
TRUTH = os.environ.get("CLIC_DATA", "/cmsuf/data/store/user/mmazza/hepattn_clic_data") + "/test_clic_common_raw.root"

BRANCHES = os.environ.get("CONVENTIONS", "mpflow_proxy").split(",")
REF = BASELINE.label
COLORS = {a.label: a.color for a in ARMS}
LINEWIDTHS = {a.label: (2.4 if a.heavy else 1.5) for a in ARMS}
MARKERS = {a.label: a.marker for a in ARMS}

missing = [f"  {a.label} -> {a.root}" for a in ARMS if not a.root.exists()]
if missing:
    raise SystemExit("evaluation .root missing, so this set cannot be drawn:\n" + "\n".join(missing))
networks = [{"name": f"{a.label} [{br}]", "path": str(a.root), "network_type": br, "ind_threshold": 0.65} for a in ARMS for br in BRANCHES]

# Jet clustering is the expensive step: cache the per-jet residuals, keyed to the networks, since the shared-event
# intersection changes with the set.
_key = hashlib.sha256("|".join(sorted(f"{n['name']}={n['path']}" for n in networks)).encode()).hexdigest()[:12]
CACHE = CACHE_DIR / f"{PREFIX}_residuals_{len(networks)}net_{_key}.npz"
if CACHE.exists() and not os.environ.get("REBUILD_PERF_CACHE"):
    print(f"Loading cached residuals from {CACHE}")
    _z = np.load(CACHE, allow_pickle=True)
    RES = {k: _z[k].item() for k in _z.files}
else:
    perf = Performance(PerformanceConfig.from_dict({"truth_path": TRUTH, "networks": networks}))
    perf.reorder_and_find_intersection()
    perf.compute_jets(n_procs=16)
    perf.hung_match_jets()
    perf.compute_event_features()
    perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)
    RES = {
        name: {"ref_e": np.asarray(perf.data[name]["jet_residuals"]["ref_e"]), "e_rel": np.asarray(perf.data[name]["jet_residuals"]["e_rel"])}
        for name in perf.data
        if "jet_residuals" in perf.data[name]
    }
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(CACHE, **{k: np.array(v, dtype=object) for k, v in RES.items()})
    print(f"Cached residuals to {CACHE}")

e_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
mids = (e_bins[:-1] + e_bins[1:]) / 2
LO, HI = 1, 8  # 20-40 GeV and 160-180 GeV


def curve(name, idx=None):
    """Per-bin median and IQR of the jet-E response, plus the global IQR and median. `idx` = bootstrap draw."""
    ref_e, e_rel = RES[name]["ref_e"], RES[name]["e_rel"]
    if idx is not None:
        ref_e, e_rel = ref_e[idx], e_rel[idx]
    med = np.full(len(mids), np.nan)
    iqr = np.full(len(mids), np.nan)
    for i, (a, b) in enumerate(pairwise(e_bins)):
        m = (ref_e > a) & (ref_e < b)
        if m.sum() == 0:
            continue
        v = e_rel[m]
        med[i] = np.percentile(v, 50)
        iqr[i] = np.percentile(v, 75) - np.percentile(v, 25)
    return med, iqr, np.percentile(e_rel, 75) - np.percentile(e_rel, 25), np.percentile(e_rel, 50)


def bootstrap(name, n=400, seed=0):
    """sigma_stat on the per-bin IQR, the global IQR and the global median, by resampling matched jets."""
    rng = np.random.default_rng(seed)
    n_jets = len(RES[name]["ref_e"])
    draws = [curve(name, rng.integers(0, n_jets, n_jets)) for _ in range(n)]
    return np.stack([d[1] for d in draws]), np.array([d[2] for d in draws]), np.array([d[3] for d in draws])


results, boots, globals_, medians = {}, {}, {}, {}
for name in [PANDORA, *(f"{a.label} [{br}]" for br in BRANCHES for a in ARMS)]:
    if name not in RES:
        continue
    med, iqr, glob, gmed = curve(name)
    biqr, bglob, bgmed = bootstrap(name)
    results[name], boots[name] = (med, iqr), biqr
    globals_[name], medians[name] = (glob, bglob.std()), (gmed, bgmed.std())

ref0 = RES[f"{REF} [{BRANCHES[0]}]"]["ref_e"]
print(f"\njets per truth-E bin, for {REF} [{BRANCHES[0]}]:")
print("  " + "  ".join(f"E{int(m)}:{int(((ref0 > a) & (ref0 < b)).sum())}" for m, (a, b) in zip(mids, pairwise(e_bins), strict=True)))
_counts = {k: len(v["ref_e"]) for k, v in RES.items()}
print(f"  total matched jets: {len(ref0)}   (across all arms: {min(_counts.values())}-{max(_counts.values())})")

# One row per convention: median (scale) and IQR (resolution) side by side, Pandora as the grey backdrop.
FIG.mkdir(exist_ok=True)
fig, axes = plt.subplots(len(BRANCHES), 2, figsize=(13, 4.2 * len(BRANCHES)), constrained_layout=True, squeeze=False)
for (m_ax, i_ax), br in zip(axes, BRANCHES, strict=True):
    if PANDORA in results:
        m_ax.plot(mids, results[PANDORA][0], label="Pandora", **PANDORA_STYLE)
        i_ax.errorbar(mids, results[PANDORA][1], yerr=boots[PANDORA].std(axis=0), label="Pandora", capsize=2, **PANDORA_STYLE)
    for a in ARMS:
        nm = f"{a.label} [{br}]"
        med, iqr = results[nm]
        st = {"color": COLORS[a.label], "lw": LINEWIDTHS[a.label], "markersize": 4}
        m_ax.plot(mids, med, MARKERS[a.label] + "-", label=a.label, **st)
        i_ax.errorbar(mids, iqr, yerr=boots[nm].std(axis=0), fmt=MARKERS[a.label] + "-", label=a.label, capsize=2, **st)
    m_ax.axhline(0, ls="--", color="k", alpha=0.4)
    m_ax.set(xlabel="Jet $E^{truth}$ [GeV]", ylabel="Jet median$((E^{reco}-E^{truth})/E^{truth})$", title=f"Median response — {br}")
    i_ax.set(xlabel="Jet $E^{truth}$ [GeV]", ylabel="Jet IQR$((E^{reco}-E^{truth})/E^{truth})$", title=f"IQR — {br}")
    for ax in (m_ax, i_ax):
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
fig.suptitle(TITLE, fontsize=10)
out = FIG / f"{PREFIX}_{'_'.join(BRANCHES)}_jet_median_iqr.png"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")

# The scan's own axis: global jet-E IQR and median against decoder k, one line per convention. Only for a set with
# one arm per k: a set that mixes training settings has no single line to draw.
ks = [a.k for a in ARMS]
if len(set(ks)) == len(ks):
    fig, (i_ax, m_ax) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for br, ls in zip(BRANCHES, ["-", "--"], strict=False):
        for ax, store in ((i_ax, globals_), (m_ax, medians)):
            vals = [store[f"{a.label} [{br}]"] for a in ARMS]
            ax.errorbar(ks, [v[0] for v in vals], yerr=[v[1] for v in vals], fmt="o" + ls, color="#003f5c", capsize=3, label=br)
    for ax, store, ylab in (
        (i_ax, globals_, "global jet IQR$(E^{reco}/E^{truth})$"),
        (m_ax, medians, "global jet median$((E^{reco}-E^{truth})/E^{truth})$"),
    ):
        if PANDORA in store:
            ax.axhline(store[PANDORA][0], color="#888888", lw=2, label="Pandora")
        ax.set(xscale="log", xlabel="decoder linformer_k (encoder k = 64)", ylabel=ylab)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    out = FIG / f"{PREFIX}_{'_'.join(BRANCHES)}_jet_iqr_vs_k.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

# Per convention: the per-bin IQR above, and each arm's IQR divided by Pandora's below. Pandora is dashed on top and
# becomes the dashed line at 1 in the ratio. The ratio's error bar is the spread of the per-draw ratio of the two
# bootstraps, which resample each network's matched jets independently.
if PANDORA in results:
    pandora_dashed = PANDORA_STYLE | {"ls": "--"}
    for br in BRANCHES:
        fig, (i_ax, r_ax) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05}, constrained_layout=True
        )
        i_ax.errorbar(mids, results[PANDORA][1], yerr=boots[PANDORA].std(axis=0), label="Pandora", capsize=2, **pandora_dashed)
        r_ax.axhline(1, **pandora_dashed)
        for a in ARMS:
            nm = f"{a.label} [{br}]"
            st = {"color": COLORS[a.label], "lw": LINEWIDTHS[a.label], "markersize": 4, "capsize": 2}
            fmt = MARKERS[a.label] + "-"
            i_ax.errorbar(mids, results[nm][1], yerr=boots[nm].std(axis=0), fmt=fmt, label=a.label, **st)
            ratio = results[nm][1] / results[PANDORA][1]
            r_ax.errorbar(mids, ratio, yerr=(boots[nm] / boots[PANDORA]).std(axis=0), fmt=fmt, **st)
        i_ax.set(ylabel="Jet IQR$((E^{reco}-E^{truth})/E^{truth})$", title=f"IQR — {br}")
        i_ax.legend(fontsize=8, ncol=2)
        r_ax.set(xlabel="Jet $E^{truth}$ [GeV]", ylabel="IQR / Pandora")
        for ax in (i_ax, r_ax):
            ax.grid(alpha=0.3)
        out = FIG / f"{PREFIX}_{br}_jet_iqr_ratio_pandora.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")

print("\nglobal jet-E median response (+- bootstrap sigma_stat):")
for name, (g, e) in medians.items():
    print(f"  {name:<40} {g:+.4f} +- {e:.4f}")
print("\nglobal jet-E IQR (+- bootstrap sigma_stat):")
for name, (g, e) in globals_.items():
    print(f"  {name:<40} {g:.4f} +- {e:.4f}")
print("\nper-bin IQR:")
print(f"{'arm':>40} " + " ".join(f"E{int(m):>5}" for m in mids))
for name, (_med, iqr) in results.items():
    print(f"{name:>40} " + " ".join(f"{x:.3f}" for x in iqr))

# Same events for every arm, so resampling them independently overstates sigma_stat; sigma_repro is not measured.
print("\narm - reference, per convention (sigma_stat only, no verdicts):")
for br in BRANCHES:
    nr = f"{REF} [{br}]"
    print(f"\n  --- {br} ---")
    for a in ARMS[1:]:
        na = f"{a.label} [{br}]"
        for lbl, i in [("IQR @ 0-20 (low E)", 0), ("IQR @ 20-40", LO), ("IQR @ 160-180 (high E)", HI)]:
            d = boots[na][:, i] - boots[nr][:, i]
            print(f"    {a.label:<22} {lbl:<24} {results[na][1][i] - results[nr][1][i]:+.4f} +- {d.std():.4f}")
        for lbl, store in (("IQR global", globals_), ("median global (scale)", medians)):
            print(f"    {a.label:<22} {lbl:<24} {store[na][0] - store[nr][0]:+.4f} +- {np.hypot(store[na][1], store[nr][1]):.4f}")

if len(BRANCHES) == 2:
    print("\nsign check (both conventions must agree before a difference is believed):")
    for a in ARMS[1:]:
        ds = [globals_[f"{a.label} [{br}]"][0] - globals_[f"{REF} [{br}]"][0] for br in BRANCHES]
        agree = "same sign" if ds[0] * ds[1] > 0 else "OPPOSITE signs -> reads as noise"
        print(f"  IQR global  {a.label:<22} {BRANCHES[0]} {ds[0]:+.4f}   {BRANCHES[1]} {ds[1]:+.4f}   {agree}")

br = BRANCHES[0]
gr = globals_[f"{REF} [{br}]"][0]
print(f"\nsummary — decoder k against global jet-E IQR ({br}):")
print(f"  {'arm':<22} {'dec k':>6} {'params':>10} {'IQR':>8} {'dIQR':>8} {'sigma_stat':>11}")
for a in ARMS:
    g, e = globals_[f"{a.label} [{br}]"]
    print(f"  {a.label:<22} {a.k:>6} {a.params:>10,} {g:>8.4f} {g - gr:>+8.4f} {e:>11.4f}")
if PANDORA in globals_:
    g, e = globals_[PANDORA]
    print(f"  {'Pandora':<22} {'--':>6} {'--':>10} {g:>8.4f} {g - gr:>+8.4f} {e:>11.4f}   (classical, not an arm)")
