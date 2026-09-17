"""Task 1: overlay the 700k v7 runs' Jet-E response median + IQR against the v6 family +
Pandora, to test whether the inverted high-E IQR trend depends on model capacity, and as a
triple-check that the two v7 trainings (1-node + 2-node) agree.
Adapted from studies/glow_jet_iqr/00_cross_run/compare_runs_iqr.py.

Auto-discovers each run's eval ROOT (ckpts/*__test.root), so it plots whichever runs have
been evaluated and SKIPs the rest. Re-run as more evals land.
Invoke: pixi run -e clic python studies/glow_jet_iqr/01_small_model_v7/plot_v7_iqr.py

v7 run -> folder -> slurm job (verified via each folder's metadata.yaml slurm_job_id):
  2-node (36450143, FINISHED, best epoch 192) -> clic_v7_20260706-T181417
  1-node (36472892, running->eval when done)  -> clic_v7_20260706-T181418
"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn/src")
from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig

L = "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/logs"
OUT = "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/glow_jet_iqr/01_small_model_v7"

# (display name, run folder) — eval ROOT auto-discovered as ckpts/*__test.root
RUNS = [
    ("3xL4 v6 (plotted)", "clic_v6_20260605-T113014"),
    ("1xB200 v6",         "clic_v6_20260606-T000306"),
    ("6xL4 v6",           "clic_v6_20260605-T143447"),
    ("4xB200 v6",         "clic_v6_20260612-T102707"),
    ("v7 700k (2-node)",  "clic_v7_20260706-T181417"),
    ("v7 700k (1-node)",  "clic_v7_20260706-T181418"),
]

networks = []
for disp, folder in RUNS:
    hits = sorted(glob.glob(f"{L}/{folder}/ckpts/*__test.root"))
    if hits:
        networks.append({"name": disp, "path": hits[0], "network_type": "mpflow", "ind_threshold": 0.65})
        print(f"USING: {disp} -> {hits[0]}")
    else:
        print(f"SKIP (no eval root yet): {disp} -> {L}/{folder}/ckpts/*__test.root")

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
# v6 family muted grey; v7 highlighted
colors = {
    "3xL4 v6 (plotted)": "#ffa600", "1xB200 v6": "#ef5675",
    "6xL4 v6": "#7a5195", "4xB200 v6": "#008080",
    "v7 700k (2-node)": "#000000", "v7 700k (1-node)": "#5b5b5b",
}
print(f"\n{'run':>18} " + " ".join(f"E{int(m):>3}" for m in mids))
for name in perf.network_names:
    if name == "Pandora":
        continue
    med, iqr = curve(name)
    print(f"{name:>18} IQR " + " ".join(f"{x:.3f}" for x in iqr))
    c = colors.get(name, None)
    lw = 2.4 if name.startswith("v7") else 1.2
    a1.plot(mids, med, "o-", label=name, color=c, markersize=3, linewidth=lw)
    a2.plot(mids, iqr, "o-", label=name, color=c, markersize=3, linewidth=lw)

if "Pandora" in perf.network_names:
    medp, iqrp = curve("Pandora")
    print(f"{'Pandora':>18} IQR " + " ".join(f"{x:.3f}" for x in iqrp))
    a1.plot(mids, medp, "s--", label="Pandora", color="#003f5c", markersize=3)
    a2.plot(mids, iqrp, "s--", label="Pandora", color="#003f5c", markersize=3)

a1.axhline(0, ls="--", color="k", alpha=0.4)
a1.set_xlabel("Jet truth E [GeV]"); a1.set_ylabel("Median jet E response"); a1.set_title("Jet E response median"); a1.legend(fontsize=8)
a2.set_xlabel("Jet truth E [GeV]"); a2.set_ylabel("IQR of jet E response"); a2.set_title("Jet E response IQR"); a2.legend(fontsize=8)
fig.suptitle("CLIC Glow: 700k v7 (2-node) vs v6 family vs Pandora — Jet E response")
fig.savefig(f"{OUT}/jet_iqr_v7_vs_v6.png", dpi=130)
print(f"\nSaved: {OUT}/jet_iqr_v7_vs_v6.png")
