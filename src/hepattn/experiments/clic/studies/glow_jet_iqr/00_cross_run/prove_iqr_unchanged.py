"""Run the full CLIC jet-response pipeline on BOTH the original and neutral-pt-fixed
ROOT files (against the same truth) and compare the Jet IQR-vs-energy curves.

Replicates notebook cells 4-16. Saves an overlay plot + prints the IQR arrays.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn/src")
from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig

CKPT = "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/logs/clic_v6_20260605-T113014/ckpts"
OUT = "/tmp/claude-9184/-blue-avery-m-mazza-projects-fastml-hepattn/82c52808-a08b-4705-8d34-7d9c726d2bcb/scratchpad"

config_dict = {
    "truth_path": "/blue/avery/m.mazza/projects/fastml/hepattn/data/clic/test_clic_common_raw.root",
    "networks": [
        {"name": "old_nobug_applied", "path": f"{CKPT}/epoch=195-val_loss=3.78800__test.root",
         "network_type": "mpflow", "ind_threshold": 0.65},
        {"name": "neutralptfix", "path": f"{CKPT}/epoch=195-val_loss=3.78800__test_neutralptfix.root",
         "network_type": "mpflow", "ind_threshold": 0.65},
    ],
}

config = PerformanceConfig.from_dict(config_dict)
perf = Performance(config)
perf.reorder_and_find_intersection()
perf.compute_jets(n_procs=20)
perf.hung_match_jets()
perf.compute_event_features()
perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)

pt_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
mids = (pt_bins[:-1] + pt_bins[1:]) / 2
lo, hi = pt_bins[:-1], pt_bins[1:]

def iqr_curve(name):
    res = perf.data[name]["jet_residuals"]
    med = np.full(len(mids), np.nan); iqr = np.full(len(mids), np.nan); ns = np.zeros(len(mids))
    for i, (a, b) in enumerate(zip(lo, hi)):
        m = (res["ref_e"] > a) & (res["ref_e"] < b)
        if m.sum() == 0:
            continue
        v = res["e_rel"][m]
        med[i] = np.percentile(v, 50)
        iqr[i] = np.percentile(v, 75) - np.percentile(v, 25)
        ns[i] = m.sum()
    return med, iqr, ns

med_o, iqr_o, n_o = iqr_curve("old_nobug_applied")
med_n, iqr_n, n_n = iqr_curve("neutralptfix")

print("\n=== Jet energy-response IQR vs truth jet E ===")
print(f"{'E bin':>12} {'N':>7} {'IQR old':>10} {'IQR fix':>10} {'Δ':>10} {'med old':>9} {'med fix':>9}")
for i in range(len(mids)):
    print(f"[{lo[i]:3.0f},{hi[i]:3.0f}) {int(n_o[i]):>7} {iqr_o[i]:>10.5f} {iqr_n[i]:>10.5f} "
          f"{iqr_n[i]-iqr_o[i]:>10.2e} {med_o[i]:>9.4f} {med_n[i]:>9.4f}")
print(f"\nmax |ΔIQR| = {np.nanmax(np.abs(iqr_n-iqr_o)):.3e}   max |Δmedian| = {np.nanmax(np.abs(med_n-med_o)):.3e}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
a1.plot(mids, med_o, "o-", label="original (no-op bug)", color="#ffa600")
a1.plot(mids, med_n, "s--", label="neutral-pt fixed", color="#003f5c", markersize=4)
a1.axhline(0, ls="--", color="k", alpha=0.4); a1.set_xlabel("Jet truth E [GeV]"); a1.set_ylabel("Median jet E response"); a1.legend(); a1.set_title("Jet E response median")
a2.plot(mids, iqr_o, "o-", label="original (no-op bug)", color="#ffa600")
a2.plot(mids, iqr_n, "s--", label="neutral-pt fixed", color="#003f5c", markersize=4)
a2.set_xlabel("Jet truth E [GeV]"); a2.set_ylabel("IQR of jet E response"); a2.legend(); a2.set_title("Jet E response IQR (the disputed plot)")
fig.suptitle("Original vs neutral-pt-fixed  —  3x L4 run, epoch 195")
fig.savefig(f"{OUT}/jet_iqr_old_vs_fixed.png", dpi=130)
print(f"\nSaved plot: {OUT}/jet_iqr_old_vs_fixed.png")
