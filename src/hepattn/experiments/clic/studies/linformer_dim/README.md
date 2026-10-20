# Decoder-only Linformer `k` scan

Shrinking `linformer_k` in the encoder degrades the encoder loss terms, so this scan shrinks it in the decoder only.
Starting from `linformer_size/configs/small_ref_linformer.yaml`, the decoder's `linformer_k` is halved from 64 to 4,
and the encoder stays at 64. One value sets all three decoder attentions: query←hits cross-attention (`q_ca`), query
self-attention (`q_sa`) and hits←query cross-attention (`kv_ca`).

All runs: 1 B200, global batch 2048, GPU matcher. Two training settings, each a full scan:

| set (`ARM_SET`) | peak LR | weight decay | epochs |
|---|---|---|---|
| `scan` | 3.2e-4 | 1e-4 | 200 |
| `scan_wd0p1` | 6.2e-4 | 0.1 | 400 |

`settings` draws the two settings' `k`=64 and `k`=4 runs side by side.

## Reproducing

From `src/hepattn/experiments/clic`:

```shell
python studies/linformer_dim/make_configs.py studies/linformer_size/configs/small_ref_linformer.yaml studies/linformer_dim/configs
TRAIN=$(sbatch --parsable submit_training_b200.sh studies/linformer_dim/configs/small_linformer_deck64.yaml)   # --time=26:00:00 for 400 epochs
sbatch --dependency=afterok:$TRAIN --export=ALL,TRAIN_JOB=$TRAIN,EVAL_CONFIG=configs/eval_linformer.yaml submit_eval_l4.sh
sbatch --export=ALL,ARM_SET=scan studies/linformer_dim/submit_plots.sh studies/linformer_dim/plot_jet_iqr.py studies/linformer_dim/plot_performance.py
```

`arms.py` lists the trained runs (under mmazza's `/cmsuf` logs) and their lowest-`val_loss` checkpoints. The plot
scripts draw the `performance.ipynb` set plus jet-E median/IQR, IQR against `k`, and IQR relative to Pandora, for the
`mpflow_proxy` convention by default (`CONVENTIONS=mpflow,mpflow_proxy` for both). Figures and pipeline caches are
written to `figures/` and `cache/`, neither tracked.

## Results

Global jet-E IQR, `mpflow_proxy`, ± bootstrap error on the test sample:

| decoder `k` | params | `scan` | `scan_wd0p1` |
|---|---|---|---|
| 64 | 1.19M | 0.0631 ± 0.0005 | 0.0613 ± 0.0005 |
| 32 | 1.07M | 0.0667 ± 0.0005 | 0.0609 ± 0.0005 |
| 16 | 1.01M | 0.0698 ± 0.0006 | 0.0648 ± 0.0005 |
| 8 | 979k | 0.0716 ± 0.0007 | 0.0636 ± 0.0005 |
| 4 | 964k | 0.0733 ± 0.0006 | 0.0677 ± 0.0006 |
| Pandora | | 0.0628 ± 0.0004 | 0.0628 ± 0.0004 |

- Under `scan`, every halving of `k` costs resolution, mostly above 100 GeV and in photon angles; efficiency,
  fake rate and energy scale barely move.
- `scan_wd0p1` improves every arm and halves the cost of shrinking `k`: its `k`=64 beats Pandora overall, `k`=32
  matches `k`=64, and `k`=4 loses +0.0063 instead of +0.0103.
- The error bars cover the finite test sample only; the spread between two trainings of one config is not measured.
  `k`=8 beating `k`=16 in `scan_wd0p1` suggests it is of order 0.001–0.002, so read the table as an ordering.
- `val_loss` does not track these numbers (e.g. `k`=4 has a lower `val_loss` than `k`=8 and 16 in `scan`).
