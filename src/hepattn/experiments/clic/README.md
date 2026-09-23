# Glow: Particle Flow with CLIC

This work is described in our preprint: [GLOW: A Unified Transformer for Diverse Reconstruction Tasks in Particle Physics](https://arxiv.org/abs/2508.20092)

This branch (`clic-paper-main`) continues from the code of that paper (tag `clic-paper`).

## Running the model

Everything for CLIC runs in the `clic` pixi environment. It contains the whole GPU
training stack (torch with CUDA, flash-attention, the compiled `lap1015` solver) plus
the analysis packages (`fastjet`, `energyflow`, `vector`, `pathos`), so training,
evaluation and the performance notebooks all use this one environment. There is no
need to install the `default` environment described in the top level
[README.md](../../../../README.md); that one serves the other experiments and lacks the
analysis packages.

Clone the repository, pull the pixi container (see the top level README for the image),
install and activate the `clic` environment:

```shell
git clone git@github.com:mmfsz/hepattn.git -b clic-paper-main
cd hepattn
apptainer pull pixi.sif docker://ghcr.io/prefix-dev/pixi:0.54.1-jammy-cuda-12.8.1
apptainer shell --nv --bind /blue/,/cmsuf/ pixi.sif
pixi install -e clic --locked
pixi shell -e clic
cd src/hepattn/experiments/clic/
```

The install takes a while the first time (the environment is about 15 GB). The
container is only needed on systems whose `libc` is older than 2.28; on HiPerGator it is
the supported way to run, and the submit scripts use it.

Check the install once (both must print `True`; the first needs a rebuild with
`pixi reinstall hepattn` if it does not, see the top level README):

```shell
python -c "import lap1015; print(lap1015.releases_gil)"
python -c "import flash_attn, torch; print(torch.cuda.is_available() or 'no GPU here, fine on a login node')"
```

The GPU matching solver (`configs/matcher_jv.yaml`) needs one more build, described in
the top level README under "solving the matching on the GPU":
`pixi run -e clic bash setup/build_torch_linear_assignment.sh`.

On UF HiPerGator see [README_HPG.md](./README_HPG.md) for the submit scripts and the data
location; the data paths live in `configs/hpg.yaml`, which the scripts layer over any
model config. To train interactively:

```shell
python main.py fit --config configs/base.yaml --config configs/hpg.yaml --trainer.devices=1
```

## Evaluation

To evaluate the model you need to run the following command:

```shell
python main.py test \
    -c <path to config.yaml> \
    --data.test_path test_clic_common_infer.root \
    --data.is_inference true \
    --trainer.precision 32-true \
    --matmutl_precision highest
```

- Flags `--data.is_inference true` and `--trainer.precision 32-true` are important for correct evaluation of the model performance.
- **Don't forget to change the attention type to `torch` in the config file.**
- **You may also need to remove the compile callback if present in the config file.**


To start a notebook on a compute node:

```shell
jupyter notebook --no-browser --ip=0.0.0.0 --port 8888
```

## CLIC Data

At UCL, files are available on `plus1` under `/unix/atlastracking/svanstroud/dmitrii_clic`, and also on `hypatia` under `/share/gpu1/syw24/dmitrii_clic`.

| File Name | Purpose / Usage | Preprocessing Applied | Notes / Details |
| :------------------------------ | :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_clic` | For **training** the model. | "Train-like" preprocessing | Applies cuts on tracks, topoclusters, and truth particles; creates target incidence matrix. |
| `val_clic` | For **validation** during model development. | "Train-like" preprocessing | Applies cuts on tracks, topoclusters, and truth particles; creates target incidence matrix. |
| `test_clic_raw.root` | For **performance evaluation**. | **None** ("raw" file) | - |
| `test_clic_fix.root` | Used by MPflow to compare preprocessed targets with model predictions. | "Train-like" preprocessing | Applies cuts on tracks, topoclusters, and truth particles; creates target incidence matrix. |
| `test_clic_common_raw.root` | For **performance evaluation**. | **None** ("raw" file) | Contains the **same events as Nilotpal's evaluation**. |
| `test_clic_common_infer.root` | Evaluates the **real performance** of the model during inference. | "Infer-like" preprocessing | Does not apply cuts; converts CLIC format, removes unused variables, correctly defines truth particles. Should be launched with `data.is_inference true` flag. Contains the **same events as Nilotpal's evaluation**. |

**Definition of Preprocessing Types:**

* **"Train-like"**: Applies cuts on tracks, topoclusters, and truth particles, and creates target incidence matrix.
* **"Infer-like"**: No cuts applied; converts CLIC format, removes unused variables, and correctly defines truth particles.
* **"Raw"**: Original CLIC files with correctly defined truth particles.

**"Truth Particles"**: Refer to Section 5.1 in [https://arxiv.org/pdf/2410.23236](https://arxiv.org/pdf/2410.23236).