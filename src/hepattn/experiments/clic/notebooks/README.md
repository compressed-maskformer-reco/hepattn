# CLIC performance notebooks

Notebooks for evaluating the trained particle-flow model and comparing it to the
CLIC baseline (Pandora) and other reconstruction algorithms. The main notebook is
`performance.ipynb`.

For the full model/architecture walkthrough, see the
[CLIC explainer](../EXPLAINER.md). This README defines the key terms used in the
notebooks. *(More to be added later.)*

## Definitions

- **Reconstruction algorithm** — a method that turns detector inputs into a list
  of particles. The ones compared here are our model (**mpflow**), **Pandora**
  (the standard CLIC particle-flow algorithm, used as the benchmark baseline),
  **HGPFlow**, and **MLPF**.

- **Truth particles** — the generator-level particles that actually exist in the
  event (with fiducial cuts applied). This is the single shared reference every
  algorithm is scored against. Read from the raw file (see below).

- **`infer` file vs `raw` file** — two test files covering the *same* 20,000
  events, joined by **event number**:
  - `test_clic_common_infer.root` (`EventTree`) — feature-engineered, padded
    inputs fed to the **model** during `main.py test`. Contains no truth. Produces
    the model's prediction file.
  - `test_clic_common_raw.root` (`events`) — the raw file, read **only by the
    notebook**. Supplies both the **truth particle lists** and the **Pandora
    reconstruction**.
  The notebook loads the model's **prediction file** plus the **raw file** — never
  the infer file directly.

- **Evaluation-time matching** — the notebook's matching of *reconstructed
  objects to truth objects*, done purely to compute metrics. Inference emits only
  an unordered particle list with no truth correspondence, so this association
  must be built here — identically for every algorithm — so all are scored fairly
  against the same truth. This is **not** the training-time Hungarian match (which
  assigns query slots to compute the loss and never runs at test time). See the
  explainer's [Step 2](../EXPLAINER.md#step-2-run-performance-notebook).
  - **Particle matching** — reco particles ↔ truth particles (charged and neutral
    separately), using a combined ΔR + Δpₜ cost.
  - **Jet matching** — reco particles and truth particles are first *clustered*
    into jets (anti-kₜ, R = 0.5, FastJet), then reco jets ↔ truth jets are matched
    by ΔR. Jets do not exist in the raw output; they are built here.

- **Residual** — the per-object comparison `(reco − truth) / truth` for pₜ, energy,
  η, φ. The basis for resolution plots.

- **Efficiency / fake rate** — efficiency = fraction of truth particles that got a
  matched reco particle; fake rate = fraction of reco particles with no truth
  match. Typically plotted vs pₜ.

- **`mpflow` vs `mpflow_proxy`** (the `network_type` in the notebook config):
  - `mpflow` — evaluates the **neural-network regression output** (`pred_{var}`),
    the model's learned prediction after correcting the proxy.
  - `mpflow_proxy` — evaluates only the **proxy kinematics** (`proxy_{var}`), the
    physics-based incidence-weighted estimate with no regression correction (a
    simple baseline that bypasses the regression head).
