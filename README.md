# hepattn

We present a general end-to-end ML approach for particle physics reconstruction by adapting cutting-edge object detection techniques.
We demonstrate that a single encoder-decoder transformer can solve many different reconstruction problems that traditionally required specialised, task-specific approaches for each domain.

This general approach has been applied to various reconstruction tasks and detector setups:

- **Pixel cluster splitting** - ATLAS [[Internal][tide]]
- **Hit filtering** - TrackML [[arXiv][trackml]], ITk [WIP]
- **Tracking** - TrackML [[arXiv][trackml]], ATLAS [[Internal][tide]]
- **Primary vertexing** - *Interested in working on this? Get in touch!*
- **Secondary vertexing** - Delphes [[EPJC][vertexing]]
- **Particle flow** - [WIP]

[tide]: https://indico.cern.ch/event/1550297/contributions/6559827/
[trackml]: https://arxiv.org/abs/2411.07149
[vertexing]: https://link.springer.com/article/10.1140/epjc/s10052-024-13374-5

## ✨ Key Features

- **🏗️ Modular architecture**: Encoder, decoder, and task modules for flexible experimentation
- **⚡ Efficient attention**: Seamlessly switch between torch SDPA, FlashAttention, and FlexAttention
- **🔬 Cutting-edge transformers**: HybridNorm, LayerScale, value residuals, register tokens, local attention
- **🚀 Performance optimised**: Full `torch.compile` and nested tensor support
- **🧪 Thoroughly tested**: Comprehensive tests across multiple reconstruction tasks
- **📦 Easy deployment**: Packaged with Pixi for reproducible environments


## 🛠️ Setup

First clone the repository:

```shell
git clone git@github.com:samvanstroud/hepattn.git
cd hepattn
```

We recommend using a container to set up and run the code.
This is necessary if your system's `libc` version is `<2.28`
due to requirements of recent `torch` versions.
We use `pixi`'s CUDA image, which you can access with:

```shell
apptainer pull pixi.sif docker://ghcr.io/prefix-dev/pixi:0.54.1-jammy-cuda-12.8.1
apptainer shell --nv pixi.sif
```

**📝 Note**: If you are not using the `pixi` container, you will need to make sure
`pixi` is installed according to https://pixi.sh/latest/.

You can then install the project with locked dependencies:

```shell
pixi install --locked -e clic
```

**📝 Note**: The `default` environment targets GPU machines and installs FA2.
The `clic` environment is `default` plus the CLIC analysis packages (`fastjet`,
`energyflow`, `vector`, `pathos`), with the same pinned torch and flash-attention
build, so it can both train and run the jet and substructure analysis; use it for
anything CLIC. Each environment is about 15 GB, so install only the one you need.
The lock file is solved with the container's pixi (0.54.1, lock format 6); a newer
host pixi rewrites it in a format the container cannot read, so re-solve the lock
only inside the container (`apptainer exec pixi.sif pixi lock`).

### The `lap1015` Extension

The `lap1015` linear assignment solver is a C++ extension vendored in
[`src/lap1015`](src/lap1015) and compiled from that source by `pixi install`. It
must be built from *this* repository's source, and not from an older or upstream
build, because only this version releases the GIL while solving. Without that, the
threaded matcher serialises: `Matcher(parallel_solver=True, n_jobs=16)` quietly runs
sixteen threads that all queue behind each other, and the `lap1015_late` solver ends
up roughly 2x slower than `scipy`.

**Check your build in one line:**

```shell
pixi run python -c "import lap1015; print(lap1015.releases_gil)"
```

`True` is what you want, and `pixi reinstall hepattn` is how you get it.

`False` needs care: the flag is set at **compile** time, so it is `False` on any
extension that was patched in place rather than rebuilt — even one that does
release the GIL. It reports how the binary was produced, not how it behaves. A
`False` here means "this build is not reproducible from `src/lap1015`", which is
reason enough to reinstall, but it is not on its own evidence that matching is
serialised.

To find out whether the solve is *actually* threaded, time it. On a B200 node at
the CLIC geometry a serialised solve costs ~5.6 s per step against ~0.8 s with
`n_jobs=16`; anything near the latter is threaded, whatever the flag says. That
distinction cost a day of investigation once — the warning is about provenance,
the timing is about behaviour.

**To rebuild:**

```shell
pixi reinstall hepattn
```

This recompiles the extension from `src/lap1015/src/main.cpp`. A plain `pixi install`
will *not* do it if the environment already exists — pixi sees the package version
unchanged and skips it, which is how a build can sit stale for weeks across edits to
the C++ source.

Two things guard this, and are worth knowing about if you change the build:

- `strict-config = false` in [pyproject.toml](pyproject.toml) lets the build
  tolerate the `pixi-conda-environment` config-setting that pixi passes to the
  backend. Without it, scikit-build-core rejects the unknown option and
  `pixi install` fails outright on a fresh clone with
  `Unrecognized options in config-settings`.
- `tests/matching/test_solvers.py::test_lap1015_releases_gil` fails if the
  installed extension holds the GIL, so a stale build is caught by the test suite
  rather than by a warning nobody reads.

Note that scikit-build-core's `editable.rebuild = true` is *not* used, though it
looks like the obvious fix. Its import-time rebuild runs under the system `cmake`,
which cannot find the Python development headers, so it turns a stale extension
into an unimportable one. Rebuild explicitly instead.

### Optional: solving the matching on the GPU

By default the Hungarian matching runs on the host: the cost matrices are copied
device-to-host and solved across a thread pool. On a GPU that the model does not
saturate, the training step is host-bound and that copy plus solve dominates it.
For that case the `Matcher` has an opt-in device solver, an exact batched
Jonker-Volgenant (`torch-linear-assignment`) that solves in place on the GPU:

```yaml
matcher:
  class_path: hepattn.models.matcher.Matcher
  init_args:
    device_solver: jv
```

It is off by default and only worth turning on when training is host-bound. On a
GPU that is already busy the solver's own kernels cost more than the stall they
remove, so measure before adopting it. Its results are exact, as the host solvers'
are; the two are held to equal total assignment cost in
`tests/matching/test_device_solver.py`.

The backend is a CUDA extension that must be compiled against the environment's
torch, for the GPU it will run on, so it is not a pixi dependency. Build it with

```shell
pixi run bash setup/build_torch_linear_assignment.sh
```

and follow the instructions it prints (put the build on `PYTHONPATH`, with
`LD_LIBRARY_PATH` pointing at the environment's `lib`). The script sets the two
things that go wrong otherwise: `FORCE_CUDA=1`, without which a build on a
GPU-less login node silently produces a CPU-only extension, and
the CUDA architectures to build for (`TLA_CUDA_ARCHS`, default L4 and B200), since
there is no GPU there to detect and the environment's own `TORCH_CUDA_ARCH_LIST`
names an architecture torch rejects. `Matcher` checks
for both failure modes at construction and refuses a missing or CPU-only build.
The script also patches the kernel's launch geometry for Blackwell GPUs: upstream
has no block-size entry for compute capability 10, so a B200 fell back to 128
threads per block and left almost half of its SMs idle during the solve. 32 was
measured fastest there, with identical assignments.

Three callbacks in `hepattn.callbacks` instrument the matcher for exactly these
decisions: `MatcherTimer` attributes the training step's wall clock to the
matcher's buckets, `MatcherCostDump` saves one step's real cost matrices for
offline replay, and `MatcherShadow` solves every problem on both device and host
and records how often the two assignments differ.

## 🌟 Activating the Environment

To run the installed environment, use:

```shell
pixi shell
```

You can close the environment with `exit`.
See the [`pixi shell` docs](https://pixi.sh/latest/reference/cli/pixi/shell/) for more information.

## 🧪 Running Tests

Once inside the environment, if a GPU and relevant external data are available, just run:

```shell
pytest
```

To test parts of the code that don't require a GPU, run:

```shell
pytest -m 'not gpu'
```

To test parts of the code that don't require external input data, run:

```shell
pytest -m 'not requiresdata'
```

Please note that the current CI only tests the parts of the code that don't require a GPU or external input data:

```shell
pytest -m 'not gpu and not requiresdata'
```

## 🏃 Run Experiments

See experiment directories for instructions on how to run experiments.

- [TrackML Tracking](src/hepattn/experiments/trackml/)
- [CLIC Particle Flow](src/hepattn/experiments/clic/)

## 🤝 Contributing

If you would like to contribute, please lint and format code with

```shell
ruff check --fix .
ruff format .
```

You can also set up pre-commit hooks to automatically run these checks before committing:

```shell
pre-commit install
```
