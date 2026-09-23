"""Save one training step's real cost matrices, for offline replay against the device solver.

Phase 1 of the CUDA matching study validated the auction on synthetic problems: uniform-random
costs, swept over shapes and scales, exact against scipy with zero fallbacks. That result is
necessary and not sufficient. Real mask-BCE costs are far more degenerate than uniform-random
ones -- many near-equal entries, a structure the auction has no synthetic analogue for -- and
the auction's two untested weaknesses (thrashing on ties, and a round count that scales with how
competitive the problem is) are exactly the ones that only degenerate costs provoke.

So the exactness claim has to be re-tested on the real thing, and the real thing only exists
inside a training step on a GPU node. This callback lifts it out: it snapshots the tensor the
matcher is actually handed on one step and writes it to disk, so that
``bench_device_matcher.py --costs`` can replay it offline against scipy's float64 optimum as
many times as needed, without holding a queue slot.

What is written, matching the schema the benchmark's ``--costs`` reader expects:

- ``costs`` -- the raw ``[batch, num_pred, num_true]`` tensor, *before* ``_prepare_costs``
  sanitises it. Raw on purpose: the sentinel-filling and cropping is part of what is under
  test, and both solver paths do their own.
- ``object_valid_mask`` -- ``[batch, num_true]``, the per-event target counts.
- ``query_valid_mask`` -- ``[batch, num_pred]`` or absent, following the model.

At the CLIC B200 geometry the batch is 2048 x 5 decoder layers = 10,240 problems and the file
is ~920 MB, which is the point: a smaller sample would not exercise the shapes the study cares
about. Use ``max_problems`` to subsample when only the conditioning matters.
"""

import json
from pathlib import Path

import torch
from lightning import Callback

from hepattn.models.matcher import Matcher


class MatcherCostDump(Callback):
    """Dump one step's matcher inputs to a ``.pt``, then optionally stop training.

    Args:
        dump_step: Training batch index to snapshot. The default is past ``torch.compile``
            warmup; the costs themselves are steady from the first step, but a warm step keeps
            the accompanying shape statistics representative of the steady state.
        output_name: Basename for the ``.pt`` snapshot and its ``.json`` manifest, written to
            ``trainer.log_dir`` (or the working directory if there is none).
        max_problems: Keep only this many problems from the batch dimension, taken as an evenly
            spaced stride so every decoder layer stays represented. ``None`` keeps all of them.
        stop_after: Stop training once the dump is written. On by default -- the job exists to
            produce the file, and the remaining steps are queue time spent for nothing.
    """

    def __init__(
        self,
        dump_step: int = 50,
        output_name: str = "matcher_costs",
        max_problems: int | None = None,
        stop_after: bool = True,
    ):
        super().__init__()
        self.dump_step = dump_step
        self.output_name = output_name
        self.max_problems = max_problems
        self.stop_after = stop_after

        self._originals: list[tuple[Matcher, object]] = []
        self._seen_steps = 0
        self._capture = False
        self._captured: list[tuple] = []
        self._blob: dict | None = None
        self.written_path: Path | None = None

    # ------------------------------------------------------------------ capture

    def _capturing(self, fn):
        """Wrap ``Matcher.forward`` so every call on the chosen step keeps its inputs.

        ``MaskFormer.loss`` calls the matcher once per step with every decoder layer stacked
        layer-major along the batch axis, which is already the ``[layers x batch, ...]`` layout
        the replay tools expect; concatenating the step's calls keeps that true for a model that
        calls it more than once.
        """

        def wrapper(costs, object_valid_mask=None, query_valid_mask=None, *args, **kwargs):
            if self._capture and self._blob is None:
                self._captured.append(tuple(None if t is None else t.detach().cpu() for t in (costs, object_valid_mask, query_valid_mask)))
            return fn(costs, object_valid_mask, query_valid_mask, *args, **kwargs)

        return wrapper

    def _stack(self) -> dict | None:
        """Stack the step's captured calls layer-major and snapshot them."""
        if not self._captured:
            return None

        def cat(parts):
            return None if any(p is None for p in parts) else torch.cat(parts, dim=0)

        costs, object_valid_mask, query_valid_mask = (cat([call[i] for call in self._captured]) for i in range(3))
        self._captured = []
        return self._snapshot(costs, object_valid_mask, query_valid_mask)

    def _snapshot(self, costs, object_valid_mask, query_valid_mask) -> dict:
        """Detach the matcher's inputs to host memory, optionally subsampled."""
        keep = slice(None)
        if self.max_problems is not None and costs.shape[0] > self.max_problems:
            # Stride rather than truncate: the batch is [layer-major] stacked, so the first N
            # rows would all come from the first decoder layer.
            step = costs.shape[0] // self.max_problems
            keep = slice(0, step * self.max_problems, step)

        blob = {"costs": costs[keep].detach().cpu()}
        if object_valid_mask is not None:
            blob["object_valid_mask"] = object_valid_mask[keep].detach().bool().cpu()
        if query_valid_mask is not None:
            blob["query_valid_mask"] = query_valid_mask[keep].detach().bool().cpu()
        return blob

    # ------------------------------------------------------------------ lightning hooks

    def on_train_start(self, trainer, pl_module) -> None:
        if self._originals:
            return

        matchers = [m for m in pl_module.modules() if isinstance(m, Matcher)]
        if not matchers:
            print("MatcherCostDump: no Matcher found in the model; nothing will be dumped.")
            return

        for matcher in matchers:
            self._originals.append((matcher, matcher.forward))
            matcher.forward = self._capturing(matcher.forward)

        print(f"MatcherCostDump: instrumented {len(matchers)} matcher(s); will dump at step {self.dump_step}.")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._capture = bool(self._originals) and self._blob is None and self._seen_steps == self.dump_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._seen_steps += 1
        self._capture = False
        if self._blob is None:
            self._blob = self._stack()

        if self._blob is None or self.written_path is not None:
            return

        self._write(trainer)
        if self.stop_after:
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module) -> None:
        self._restore()
        if self._blob is None:
            print(f"MatcherCostDump: training ended before step {self.dump_step}; nothing was dumped.")

    def teardown(self, trainer, pl_module, stage) -> None:
        self._restore()

    def _restore(self) -> None:
        for matcher, forward in self._originals:
            matcher.forward = forward
        self._originals = []

    # ------------------------------------------------------------------ output

    def _write(self, trainer) -> None:
        out_dir = Path(getattr(trainer, "log_dir", None) or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.output_name}.pt"

        torch.save(self._blob, path)
        self.written_path = path

        manifest = self._manifest()
        (out_dir / f"{self.output_name}.json").write_text(json.dumps(manifest, indent=2))

        print("-" * 80)
        print(f"MatcherCostDump: step {self.dump_step}, wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
        print(f"  costs {manifest['shape']} {manifest['dtype']}")
        print(f"  targets per problem: min {manifest['targets_min']}  median {manifest['targets_median']}  max {manifest['targets_max']}")
        print(f"  non-finite entries: {manifest['non_finite']}")
        print(f"  replay with: python bench_device_matcher.py --costs {path}")
        print("-" * 80, flush=True)

    def _manifest(self) -> dict:
        """Shape and conditioning statistics, so the .pt can be read without loading a GB."""
        costs = self._blob["costs"]
        finite = torch.isfinite(costs)
        mask = self._blob.get("object_valid_mask")
        lengths = mask.sum(dim=1) if mask is not None else torch.full((costs.shape[0],), costs.shape[2])

        return {
            "shape": list(costs.shape),
            "dtype": str(costs.dtype),
            "dump_step": self.dump_step,
            "max_problems": self.max_problems,
            "has_query_valid_mask": "query_valid_mask" in self._blob,
            "targets_min": int(lengths.min()),
            "targets_median": int(lengths.median()),
            "targets_max": int(lengths.max()),
            "non_finite": int((~finite).sum()),
            # The spread of the finite costs: the auction's round count scales with how
            # competitive a problem is, so this is the number to compare against the synthetic
            # sweep's uniform [0, 1).
            "cost_min": float(costs[finite].min()) if finite.any() else float("nan"),
            "cost_max": float(costs[finite].max()) if finite.any() else float("nan"),
            "cost_std": float(costs[finite].float().std()) if finite.any() else float("nan"),
        }
