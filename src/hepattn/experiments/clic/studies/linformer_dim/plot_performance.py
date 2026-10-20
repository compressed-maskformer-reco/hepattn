"""The `performance.ipynb` plot set for the decoder linformer_k scan, every arm overlaid.

Ported from the paper-tag studies/linformer/plot_linformer_performance.py: the notebook's seven figures in its order,
with all arms on one canvas so that a class or a region that degrades shows as a shape against its neighbours.
Uses this repository's performance module, which the eval .root files were written for.

ONE FIGURE SET PER CONVENTION. `mpflow` uses the regression head's kinematics, `mpflow_proxy` rebuilds them from the
incidence head; they differ by more than most effects looked for, so compare only within one. The default is
`mpflow_proxy`, the convention the paper comparisons use; CONVENTIONS=mpflow,mpflow_proxy draws both.

The pipeline (jet clustering, jet and particle matching) is the whole cost and wants 16 cores, so it is cached per
convention and arm set; redrawing from the cache runs in seconds on a login node. REBUILD_PERF_CACHE=1 recomputes.

Run from the clic experiment dir (first time via submit_plots.sh):
    pixi run -e clic python studies/linformer_dim/plot_performance.py
"""

import hashlib
import os
import pickle  # noqa: S403 - reads only this script's own cache, never external input
import sys
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.clic.performance.performance import NetworkType, Performance, PerformanceConfig
from hepattn.experiments.clic.performance.plot_helper import PlotHelper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ARMS, PANDORA_COLORS, PREFIX

# The pipeline injects Pandora under this name; taken from the enum, which differs between branches.
PANDORA = NetworkType.PANDORA.value

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CACHE = OUT / "cache"
# The truth-level test sample, in the data directory the submit scripts read ($CLIC_DATA, same default).
TRUTH = os.environ.get("CLIC_DATA", "/cmsuf/data/store/user/mmazza/hepattn_clic_data") + "/test_clic_common_raw.root"

ROOTS = {a.key: a.root for a in ARMS}
COLORS = {a.key: a.color for a in ARMS}
# Hue alone does not separate overlapping step histograms; the dash pattern is a second, redundant channel.
LINESTYLES = {a.key: a.dash for a in ARMS}
LINEWIDTHS = {a.key: (2.2 if a.heavy else 1.4) for a in ARMS}
LABELS = {a.key: a.label for a in ARMS} | {PANDORA: "Pandora"}
EFF_FR_COLORS = {a.key: {"neut had": a.color, "photon": a.photon} for a in ARMS} | {PANDORA: PANDORA_COLORS}

PT_BINS = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])


def style_for(name):
    """Per-arm style, splatted into the helpers' `hist` calls."""
    d = {"histtype": "step", "linewidth": 1.4, "linestyle": "-"}
    if name in COLORS:
        d |= {"color": COLORS[name], "linestyle": LINESTYLES[name], "linewidth": LINEWIDTHS[name]}
    if name == PANDORA:
        d |= {"color": "#888888", "alpha": 0.3, "histtype": "stepfilled", "linewidth": 1, "linestyle": "-"}
    return d


def _cache_path(convention):
    """Cache file keyed to every arm's path: the shared-event intersection depends on the whole set."""
    key = hashlib.sha256("|".join(f"{a}={ROOTS[a]}" for a in sorted(ROOTS)).encode()).hexdigest()[:12]
    return CACHE / f"perf_{PREFIX}_{convention}_{key}.pkl"


def build(convention):
    """Return the analysed `Performance` for one convention, from cache when possible.

    Raises:
        SystemExit: If any arm's evaluation .root is missing; a dropped arm would shift the event intersection.
    """
    cache = _cache_path(convention)
    if cache.exists() and not os.environ.get("REBUILD_PERF_CACHE"):
        print(f"loading cached pipeline: {cache} ({cache.stat().st_size / 1e9:.2f} GB)", flush=True)
        with cache.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - our own cache, written by this script

    missing = [f"  {arm}: {path}" for arm, path in ROOTS.items() if not path.exists()]
    if missing:
        raise SystemExit("every arm is needed, missing evaluation .root:\n" + "\n".join(missing))
    networks = [{"name": arm, "path": str(path), "network_type": convention, "ind_threshold": 0.65} for arm, path in ROOTS.items()]

    perf = Performance(PerformanceConfig.from_dict({"truth_path": TRUTH, "networks": networks}))
    perf.reorder_and_find_intersection()
    perf.compute_jets(n_procs=16)
    perf.hung_match_jets()
    perf.compute_event_features()
    perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)
    perf.hung_match_particles(flatten=True, return_unmatched=True)

    CACHE.mkdir(exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(perf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cached pipeline -> {cache} ({cache.stat().st_size / 1e9:.2f} GB)", flush=True)
    return perf


def _headroom(ax, factor):
    """Grow the y-axis upward so an in-axes legend stops sitting on the data (decades on a log axis)."""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        if lo > 0 and hi > 0:
            ax.set_ylim(lo, hi * (10**factor))
    else:
        ax.set_ylim(lo, hi + factor * (hi - lo))


def _legend_contents(ax):
    """Handles and labels of an axes' existing legend; boxplot legends carry proxy patches the axes do not list."""
    leg = ax.get_legend()
    if leg is not None:
        handles = list(getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", []))
        labels = [t.get_text() for t in leg.get_texts()]
        if handles:
            return handles, labels
    return ax.get_legend_handles_labels()


def _restyle(fig, *, size=None, ncol=1, fontsize=None, headroom=0.0, bump_lw=None, shared_below=False, shared_ncol=3):
    """Post-hoc layout fixes on a helper figure, which is shared with performance.ipynb and so not restyled at source.

    `bump_lw` thickens the lines of the helpers that ignore the style's linewidth; `shared_below` replaces per-axes
    legends with one figure legend underneath, for eff_fr_purity's two classes per arm.
    """
    if size is not None:
        fig.set_size_inches(*size)

    if bump_lw:
        for ax in fig.axes:
            for ln in ax.get_lines():
                if ln.get_linewidth() <= 1.5 and ln.get_color() not in {"#888888", "#999999", "#cccccc"}:
                    ln.set_linewidth(bump_lw)

    if shared_below:
        handles, labels = [], []
        for ax in fig.axes:
            h, ls = _legend_contents(ax)
            for hh, ll in zip(h, ls, strict=False):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        # Below the figure, kept in the output by bbox_inches="tight" at save.
        if fig.get_layout_engine() is None:
            fig.subplots_adjust(bottom=0.16)
        fig.legend(handles, labels, loc="upper center", ncol=shared_ncol, fontsize=fontsize, frameon=False, bbox_to_anchor=(0.5, 0.0))
        return

    for ax in fig.axes:
        if headroom:
            _headroom(ax, headroom)
        if ax.get_legend() is None:
            continue
        handles, labels = _legend_contents(ax)
        if not handles:
            continue
        ax.get_legend().remove()
        ax.legend(handles, labels, loc="upper right", ncol=ncol, fontsize=fontsize, framealpha=0.85, borderpad=0.4, labelspacing=0.3)
    if size is not None and fig.get_layout_engine() is None:
        fig.tight_layout()


def draw(perf, convention):
    style = defaultdict(dict)
    for name in perf.network_names:
        style[name] = style_for(name)
    helper = PlotHelper(perf, style_dict=style, labels=LABELS, plot_path=str(FIG))

    fix_hist = partial(_restyle, headroom=0.16)
    fix_box = partial(_restyle, ncol=3, headroom=0.12)
    fix_resp = partial(_restyle, ncol=1, fontsize=8, headroom=0.3)
    fix_log = partial(_restyle, headroom=0.7)
    fix_eff = partial(_restyle, size=(7.6, 6.4), fontsize=8, bump_lw=1.9, shared_below=True)

    plots = [
        ("evt_res", lambda: helper.plot_evt_res(), fix_hist),
        ("jet_residuals", lambda: helper.plot_jet_residuals(pt_relative=True), fix_hist),
        ("jet_res_boxplot_pt", lambda: helper.plot_jet_res_boxplot(var="pt", bins=PT_BINS), fix_box),
        ("jet_response_e", lambda: helper.plot_jet_response(pt_bins=PT_BINS, use_energy=True), fix_resp),
        ("particle_residuals", lambda: helper.plot_residuals(pt_relative=True, log_y=True), fix_log),
        ("particle_residuals_neutrals", lambda: helper.plot_residuals_neutrals(pt_relative=True, log_y=True), fix_log),
        ("eff_fr_purity", lambda: helper.plot_eff_fr_purity(EFF_FR_COLORS), fix_eff),
    ]

    written, failed = [], []
    for stem, fn, fixup in plots:
        name = f"{PREFIX}_{convention}_{stem}.png"
        try:
            fig = fn()
        except Exception:  # noqa: BLE001 - one bad plot must not cost the whole pipeline
            print(f"  FAILED {name}", flush=True)
            traceback.print_exc()
            failed.append(name)
            continue
        # Some helpers return a Figure, some a list of them, some save themselves and return None.
        figs = fig if isinstance(fig, list | tuple) else [fig]
        for i, f in enumerate(figs):
            if not isinstance(f, plt.Figure):
                continue
            fixup(f)
            out = FIG / (name if len(figs) == 1 else name.replace(".png", f"_{i}.png"))
            f.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(f)
            written.append(out.name)
        if fig is None:
            print(f"  {name}: helper returned no Figure (it may have saved itself)", flush=True)
    return written, failed


FIG.mkdir(exist_ok=True)
all_written, all_failed = [], []
for convention in os.environ.get("CONVENTIONS", "mpflow_proxy").split(","):
    print(f"\n=== {convention} ===", flush=True)
    perf = build(convention)
    w, f = draw(perf, convention)
    all_written += w
    all_failed += f

print("\nwrote:")
for n in all_written:
    print(f"  {n}")
if all_failed:
    print("\nFAILED:")
    for n in all_failed:
        print(f"  {n}")
    sys.exit(1)
