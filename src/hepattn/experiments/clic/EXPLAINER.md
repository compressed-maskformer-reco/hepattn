# CLIC Particle Flow: Training and Evaluation Explained

## What is the task?

The goal is **particle flow reconstruction**: given raw detector measurements from a CLIC collision event, reconstruct the individual particles that produced them — their identity (type), momentum, and which detector hits they left behind.

A single event produces ~300 detector objects:
- **Tracks**: Curved paths left by charged particles in the tracking detector. Each track has 12 parameters (pt, eta, phi, d0, z0, tanlambda, omega, chi2, ndf, …).
- **Topological clusters (topos)**: Blobs of energy deposited in the calorimeter. Each has 10 parameters (energy, eta, phi, shower shapes, ECAL/HCAL fractions, …).

These are merged into a single sequence of up to 160 "nodes", each described by **27 features**. The model must group them into particles and predict each particle's class and 4-momentum.

---

## The 27 Input Features

Each node (track or topo) is described by a 27-dimensional vector:

| Group | Features | Notes |
|---|---|---|
| Kinematics | pt, eta, phi, sinphi, cosphi | Shared by tracks and topos |
| Interaction point | eta_int, phi_int, sinphi_int, cosphi_int | Tracks only; zero for topos |
| Track parameters | z0, d0, chi2, ndf, radiusofinnermosthit, tanlambda, omega | Tracks only; zero for topos |
| Calorimeter | e, rho, sigma_eta, sigma_phi, sigma_rho, energy_ecal, energy_hcal, energy_other, em_frac | Topos only; zero for tracks |
| Type flags | is_track, is_topo | Binary one-hot |

All features are normalised before entering the network (sqrt-transform on energy-like quantities, then symmetric min-max or standard normalisation), defined in `configs/clic_var_transform.yaml`.

---

## Model Architecture (MaskFormer)

The model uses a **transformer encoder–decoder** architecture in four stages:

### 1. Input embedding
Each node's 27 features are projected to a 256-dimensional embedding via a small MLP (`Dense(27 → 54 → 256)`). A **Fourier position encoder** also encodes the spatial position (eta, phi) into sinusoidal basis functions and adds it to the embedding.

### 2. Encoder (6 transformer layers)
The 160 node embeddings (padded to variable event sizes) are processed with **self-attention**: every node can attend to every other node, so a track can "see" nearby topo clusters and vice versa. This produces 160 context-aware embeddings of dimension 256.

Eight learnable **register tokens** are prepended to the sequence (and discarded after encoding) to absorb global event-level information without polluting per-node representations. These are extra learnable vectors that are concatenated to the front of the node sequence before the encoder layers and sliced off afterwards, so they never reach the decoder. During attention, every real node can read from and write to them, giving the transformer dedicated "scratch space" for global, event-level bookkeeping. Without them, the model tends to overload some real node's embedding with this global summary, corrupting its per-node representation; the registers keep the final node embeddings clean. See [Darcet et al., *Vision Transformers Need Registers*, ICLR 2024](https://arxiv.org/abs/2309.16588).

### 3. Decoder (4 transformer layers)
The decoder has 150 learnable **query vectors** — one slot per possible output particle. At each of 4 layers:
- Queries attend to the encoded nodes via **cross-attention** (each query learns which nodes to look at).
- An **attention mask** zeroes out contributions from unrelated nodes. This is the Mask2Former "masked attention" trick: each layer scores every (query, node) pair via a dot product of their embeddings — the model's current belief about which hits belong to each particle (the **hit mask**). Thresholding that belief gives a boolean mask, so each query attends only to the hits it currently claims. The mask is detached (a hard gate, no gradient through the masking) and refined layer by layer; a query whose mask is all-false is reset to attend everywhere so it is never starved of input.
- Queries attend to each other via **self-attention**.
- Nodes also attend back to queries, updating their representations.

After 4 decoder layers, each of the 150 query slots represents one predicted particle (or is predicted as null/background).

### 4. Task heads
Four parallel output heads predict different aspects simultaneously.

---

## The Four Tasks

### Task 1: Classification — "Is this slot a real particle, and what type?"
**Predicts:** 6-class label (charged hadron, electron, muon, neutral hadron, photon, null) for each of the 150 query slots.  
**Loss:** Cross-entropy, with class weights to compensate for imbalance (electrons and muons are rare).

### Task 2: Hit mask — "Which detector hits belong to this particle?"
**Predicts:** A 150×160 matrix of logits: one per (particle, node) pair.  
**Loss:** Binary cross-entropy + DICE loss on the binarised mask.  
The predicted mask from each layer is fed back as an attention mask into the next decoder layer, focusing the cross-attention on relevant hits.

### Task 3: Incidence regression — "What fraction of energy did each hit contribute?"
**Predicts:** A 150×160 soft assignment matrix (rows sum to 1 per node via softmax), encoding the fractional energy contribution from each node to each particle.  
**Loss:** KL divergence against the truth incidence matrix computed from MC truth.  
This is a softer, more informative version of the hit mask.

### Task 4: Momentum regression — "What are this particle's e, pt, eta, phi?"
**Predicts:** 5 kinematic quantities (e, pt, eta, sinphi, cosphi) per particle.  
**Loss:** L1 (mean absolute error) on normalised quantities. Weight = 10 (highest priority).

**How it works:** First, the incidence matrix (Task 3) is used as a physics-motivated starting point — the "proxy": hit kinematics are averaged, weighted by the incidence values, to get an initial estimate of the particle's momentum. The regression network then *corrects* this proxy estimate. This is why two outputs exist: `proxy_{var}` (physics-based estimate) and `pred_{var}` (neural network refinement).

---

## Hungarian Matching (training loss)

> **Two different matchings exist in this project — don't confuse them.** This section describes the **training-time** match that assigns query slots to truth particles so the *loss* can be computed. A separate **evaluation-time** match (particle- and jet-level, on physics quantities) is done later in the performance notebook purely to compute metrics — see [Step 2](#step-2-run-performance-notebook). The two are unrelated: the training match needs the truth-association info found only in the training file and never runs at inference, so it produces nothing the notebook can reuse.

The model predicts 150 particle slots, but they come out in **no fixed order** — slot 7 is not tied to any particular true particle. So we cannot simply compare prediction `i` to truth `i`; we must first decide which prediction is meant to explain which truth particle. That decision is **matching**: a one-to-one assignment between predicted slots and truth particles.

To make it, we build a cost matrix scoring every (prediction, truth) pair by how badly that prediction would explain that particle, combining all four tasks:

```
cost(pred_i, truth_j) = 2 × class_cost + 5 × mask_BCE + 1 × mask_DICE + 1 × KL + 10 × regression_L1
```

The **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`, optimal bipartite matching) then picks the one-to-one pairing that minimises the *total* cost over the event — a global optimum, not a greedy per-particle guess. Each event is independent, so the batch is solved in parallel using `n_jobs=16` threads. Matching only *finds* the assignment: it runs under `no_grad` on detached costs, so no gradient flows through the assignment itself. The predictions are then reordered so that prediction `i` aligns with truth particle `i` (unmatched slots are supervised as null/background), and all losses are computed on those aligned pairs.

Matching is redone at **every decoder layer**, not just the final one. This gives each layer its own aligned supervision signal (gradient flow to early layers via deep supervision), and it also produces a depth-wise **curriculum**: early layers have coarse embeddings, so their cost matrix is blurry and the assignment only demands getting each slot into roughly the right neighborhood; by the final layer the embeddings are sharp, the cost matrix is peaked, and the assignment demands exact hit membership, class, and momentum. A given query slot is thus supervised against a progressively tightening (easy → hard) target as it moves through the stack. Note this is an *emergent* effect of applying matching to progressively-refined representations across depth — not curriculum learning in the strict sense of a hand-designed easy-to-hard schedule over training steps.

---

## Training Configuration

| Setting | Value |
|---|---|
| Model dim | 256 |
| Encoder layers | 6 |
| Decoder layers | 4 |
| Attention heads | 16 |
| Particle slots | 150 |
| Register tokens | 8 |
| Total parameters | ~11.7M |
| Optimiser | Lion |
| Precision | bf16-mixed |
| Batch size/GPU | 2048 (B200) or 256 (L4) |
| Max epochs | 200 |
| Attention type | flash-varlen (variable-length FlashAttention) |

---

## Data Files and the Difference Between "raw" and "infer"

Three types of ROOT files are used, and they serve different purposes:

### Training/validation files: `train_clic_fix.root`, `val_clic_fix.root`
- **Tree name:** `EventTree`
- **`is_inference: false`** (set in config)
- Contains truth particle–track associations and incidence matrices needed for supervised training.
- **Class relabelling is applied:** Trackless charged particles (charged particles whose track was not reconstructed) are reclassified as their neutral equivalent (e.g., trackless charged hadron → neutral hadron), because the model can only "see" what the detector sees.

### Inference test file: `test_clic_common_infer.root`
- **Tree name:** `EventTree`  
- **`is_inference: true`** (set at evaluation time)
- No class relabelling; no truth mask. The model must predict particle identities from detector hits alone.
- Used as input to `main.py test` to produce the prediction ROOT file.
- Filtered: 276 events with >160 nodes are removed during dataset loading.

### Performance evaluation file: `test_clic_common_raw.root`
- **Tree name:** `events` (different!)
- **Never fed to the model** — this file is read only by the performance notebook.
- Contains **Pandora reconstruction** alongside truth particles. Pandora is the standard particle flow algorithm used at CLIC, and serves as the benchmark baseline.
- Provides the ground-truth particle lists (with PDG IDs, fiducial cuts applied) needed to evaluate jet and particle-level performance metrics.

**Why two separate test files?** The model needs `EventTree`-format data with proper feature engineering and padding (infer format), so the **infer file** is fed to `main.py test`, which writes out a **prediction ROOT file** — this is a separate step, not part of the notebook. The performance notebook then loads **two** files for its studies: (1) that **prediction ROOT file** (the model's outputs) and (2) the **raw file** (events format). Note the notebook never reads the infer file directly — only the predictions derived from it.

The raw file does double duty: it provides both the **truth particle lists** and the **Pandora reconstruction**. Truth is the single shared reference — the model's predictions *and* Pandora are each matched against the same truth particles (via Hungarian matching on physics quantities), which is what makes the comparison to the CLIC baseline fair. The prediction file and the raw file are aligned by **event number**; both cover the same 20,000 events, so event `n`'s predictions are paired with event `n`'s truth and Pandora.

---

## The Evaluation Pipeline

### Step 1: Run `main.py test`
The model runs on `test_clic_common_infer.root` and the `PflowPredictionWriter` callback saves predictions.

**Output HDF5** contains per-event, per-particle:
- Predicted class, hit masks, incidence matrix
- Regression output (`pred_{e,pt,eta,sinphi,cosphi}`)
- Proxy kinematics (`proxy_{...}`)

**Output ROOT** (converted from HDF5) contains per-event jagged arrays:
- `mpflow.{pt,eta,phi,class}` — model regression predictions (after thresholding `pred_ind > 0.5`)
- `proxy.{pt,eta,phi}` — proxy estimates (incidence-weighted sum of hit kinematics)
- `pred_ind` — boolean indicator of whether each slot is predicted as a real particle

### Step 2: Run performance notebook
The notebook reads `test_clic_common_raw.root` (truth + Pandora) and the model's ROOT file, aligns them by event number (`reorder_and_find_intersection`), then:

1. **Clusters jets** with anti-kT (R=0.5) using FastJet
2. **Matches jets** between truth and reco using Hungarian matching on ΔR
3. **Matches particles** (charged and neutral separately) using a combined ΔR + ΔpT cost
4. **Computes residuals**: (reco − truth)/truth for pT, energy, η, φ
5. **Plots** jet energy resolution, particle residuals, efficiency/fake rate vs pT

**Why re-match here — isn't this already done?** No. Inference (Step 1) emits only an *unordered list of reconstructed particles* per event, with **no** correspondence to truth — the infer file contains no truth, so the model has nothing to match against at test time. Pandora's output is the same: a bare particle list. To compute any residual or resolution you must first line up each reconstructed object with the true object it represents, and that association is exactly what this evaluation-time matching produces. It is deliberately done **here**, identically for every algorithm (model, Pandora, HGPFlow, MLPF), so all are scored on equal footing against the same truth — that common, uniform comparison is what "performance relative to the CLIC baseline" means. This is **not** the training-time [Hungarian match](#hungarian-matching-training-loss): that one assigns query slots to compute the loss, needs truth-association info absent from the inference file, and never runs at test time. Here the match is a physics object-to-object match (kinematics / ΔR) used purely for metrics; jets in particular don't even exist until the reco particles are clustered in step 1.

### `network_type: "mpflow"` vs `"mpflow_proxy"`

In the performance notebook config:
- `"mpflow"`: evaluates the **neural network regression output** (`pred_{var}`) — what the model learned to predict after correcting the proxy.
- `"mpflow_proxy"`: evaluates only the **proxy kinematics** (`proxy_{var}`) — the physics-based incidence-weighted estimate, without any regression correction. This is effectively a simple baseline that bypasses the regression head.

If `mpflow_proxy` plots look similar to a much smaller model, it means you're not evaluating the regression — switch to `"mpflow"` to see the full model performance.
