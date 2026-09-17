"""Compare Jet E-response median + IQR across all trained CLIC runs, against the same
truth + Pandora baseline. Runs the full notebook pipeline once with every run's eval
ROOT file as a network."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn/src")
from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig

L = "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/logs"
OUT = "/tmp/claude-9184/-blue-avery-m-mazza-projects-fastml-hepattn/82c52808-a08b-4705-8d34-7d9c726d2bcb/scratchpad"

# (display name, run folder, ckpt stem, val_loss)
RUNS = [
    ("3xL4 (plotted)", "clic_v6_20260605-T113014", "epoch=195-val_loss=3.78800", 3.788),
    ("1xB200",         "clic_v6_20260606-T000306", "epoch=199-val_loss=3.83626", 3.836),
    ("6xL4",           "clic_v6_20260605-T143447", "epoch=199-val_loss=3.92173", 3.922),
    ("4xB200",         "clic_v6_20260612-T102707", "epoch=190-val_loss=3.97507", 3.975),
]

import os
networks = []
for disp, folder, stem, vl in RUNS:
    p = f"{L}/{folder}/ckpts/{stem}__test.root"
    if os.path.exists(p):
        networks.append({"name": disp, "path": p, "network_type": "mpflow", "ind_threshold": 0.65})
    else:
        print(f"SKIP (missing): {disp} -> {p}")

config = PerformanceConfig.from_dict({
    "truth_path": "/blue/avery/m.mazza/projects/fastml/hepattn/data/clic/test_clic_common_raw.root",
    "networks": networks,
})
perf = Performance(config)
perf.reorder_and_find_intersection()
perf.compute_jets(n_procs=20)
perf.hung_match_jets()
perf.compute_event_features()
perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)

pt_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
mids = (pt_bins[:-1] + pt_bins[1:]) / 2
lo, hi = pt_bins[:-1], pt_bins[1:]

def curve(name):
    res = perf.data[name]["jet_residuals"]
    med = np.full(len(mids), np.nan); iqr = np.full(len(mids), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        m = (res["ref_e"] > a) & (res["ref_e"] < b)
        if m.sum() == 0:
            continue
        v = res["e_rel"][m]
        med[i] = np.percentile(v, 50); iqr[i] = np.percentile(v, 75) - np.percentile(v, 25)
    return med, iqr

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
colors = {"3xL4 (plotted)": "#ffa600", "1xB200": "#ef5675", "6xL4": "#7a5195", "4xB200": "#008080"}
print(f"\n{'run':>16} " + " ".join(f"E{int(m):>3}" for m in mids))
for name in perf.network_names:
    if name == "Pandora":
        continue
    med, iqr = curve(name)
    print(f"{name:>16} IQR " + " ".join(f"{x:.3f}" for x in iqr))
    c = colors.get(name, None)
    a1.plot(mids, med, "o-", label=name, color=c, markersize=3)
    a2.plot(mids, iqr, "o-", label=name, color=c, markersize=3)
# Pandora baseline
if "Pandora" in perf.network_names:
    medp, iqrp = curve("Pandora")
    print(f"{'Pandora':>16} IQR " + " ".join(f"{x:.3f}" for x in iqrp))
    a1.plot(mids, medp, "s--", label="Pandora", color="#003f5c", markersize=3)
    a2.plot(mids, iqrp, "s--", label="Pandora", color="#003f5c", markersize=3)

a1.axhline(0, ls="--", color="k", alpha=0.4)
a1.set_xlabel("Jet truth E [GeV]"); a1.set_ylabel("Median jet E response"); a1.set_title("Jet E response median"); a1.legend(fontsize=8)
a2.set_xlabel("Jet truth E [GeV]"); a2.set_ylabel("IQR of jet E response"); a2.set_title("Jet E response IQR"); a2.legend(fontsize=8)
fig.suptitle("CLIC Glow: Jet E response across all trained runs vs Pandora")
fig.savefig(f"{OUT}/jet_iqr_all_runs.png", dpi=130)
print(f"\nSaved: {OUT}/jet_iqr_all_runs.png")
