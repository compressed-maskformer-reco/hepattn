"""Attribute training-step wall clock to the Hungarian matcher.

This answers one question: what fraction of a training step does the matcher actually own?
It exists because a GPU trace cannot answer it. The profiler reports GPU-idle time, but idle
time covers every host-side cost in the step -- the matcher, the Python of the loss loop, the
optimiser, the logger -- and the decision of whether to move the solver onto the GPU turns on
the matcher's share alone.

The measurement is explicit ``torch.cuda.synchronize()``-bracketed timers rather than trace
attribution. A sync on entry to the matcher is what makes the numbers mean anything: the host
path's first blocking operation drains every kernel queued earlier in the step, so without the
leading sync the matcher's transfer appears to cost as much as all the GPU work preceding it.
With the sync, the GPU is caught up before the timer starts, and everything measured after it
is time in which the GPU has nothing to run -- which is exactly the prize a device solver
would be competing for.

Buckets, per step:

- ``prep`` -- device-side preparation (sanitising, masking padded queries, transposing and
  cropping to the largest event) plus the four-byte read of the per-event target counts, which
  is the sync that drains those kernels.
- ``dtoh`` -- the device-to-host copy of the prepared cost tensor, staged through the pinned
  buffer. This is the transfer a device solver removes outright.
- ``solve`` -- the host LAP solve.
- ``device`` -- the whole device-solver path, when ``device_solver`` is set. It has no ``dtoh``
  and no host ``solve``, so it is reported on its own.
- ``other`` -- the rest of the step, by subtraction.

The syncs serialise host and device, so absolute step times here run slightly longer than an
uninstrumented run. Read the *fractions*, and take throughput from a run without this callback.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from lightning import Callback

from hepattn.models.matcher import Matcher

# Buckets that are timed inside the matcher. "other" is derived by subtraction.
_MATCHER_BUCKETS = ("prep", "dtoh", "solve", "device")


class MatcherTimer(Callback):
    """Time the matcher against the whole training step, with explicit CUDA syncs.

    Args:
        warmup_steps: Steps to discard before recording. Must clear ``torch.compile`` warmup,
            which has historically taken tens of steps on this model.
        output_name: Basename for the ``.npz`` of per-step timings and the ``.json`` summary,
            written to ``trainer.log_dir`` (or the working directory if there is none).
        print_every: Print a running attribution every N recorded steps. 0 disables it.
    """

    def __init__(self, warmup_steps: int = 30, output_name: str = "matcher_timing", print_every: int = 0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.output_name = output_name
        self.print_every = print_every

        self._matchers: list[Matcher] = []
        self._originals: list[tuple[Matcher, dict]] = []
        self._step_bucket: dict[str, float] = dict.fromkeys(("total", *_MATCHER_BUCKETS), 0.0)
        self._records: list[dict[str, float]] = []
        self._step_start: float | None = None
        self._seen_steps = 0
        self._batch_size: int | None = None
        self._summary: dict | None = None

    # ------------------------------------------------------------------ timing helpers

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _timed(self, fn, bucket: str, sync_first: bool = False, sync_after: bool = False):
        """Wrap a bound method so its wall time accumulates into ``bucket``.

        ``sync_after`` is for work that only launches kernels and returns: without a trailing
        synchronize the bucket records the launch, and the kernel's time lands in whatever
        blocks on it next.
        """

        def wrapper(*args, **kwargs):
            if sync_first:
                self._sync()
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                if sync_after:
                    self._sync()
                self._step_bucket[bucket] += time.perf_counter() - start

        return wrapper

    # ------------------------------------------------------------------ lightning hooks

    def on_train_start(self, trainer, pl_module) -> None:
        if self._originals:
            return

        self._matchers = [m for m in pl_module.modules() if isinstance(m, Matcher)]
        if not self._matchers:
            print("MatcherTimer: no Matcher found in the model; no timings will be recorded.")
            return

        for matcher in self._matchers:
            saved = {
                "forward": matcher.forward,
                "_prepare_costs": matcher._prepare_costs,  # noqa: SLF001
                "_stage_to_host": matcher._stage_to_host,  # noqa: SLF001
                "_solve": matcher._solve,  # noqa: SLF001
                "_match_on_device": matcher._match_on_device,  # noqa: SLF001
            }
            self._originals.append((matcher, saved))

            # The leading sync belongs on forward and nowhere else: it must happen once, before
            # any of the matcher's own work, so that the kernels queued earlier in the step are
            # charged to "other" rather than to whichever bucket blocks on them first.
            matcher.forward = self._timed(saved["forward"], "total", sync_first=True)
            matcher._prepare_costs = self._timed(saved["_prepare_costs"], "prep")  # noqa: SLF001
            matcher._stage_to_host = self._timed(saved["_stage_to_host"], "dtoh")  # noqa: SLF001
            matcher._solve = self._timed(saved["_solve"], "solve")  # noqa: SLF001
            # The device solver is a single asynchronous kernel launch; time it to completion,
            # or the bucket reads ~2 ms whatever the solve costs.
            matcher._match_on_device = self._timed(saved["_match_on_device"], "device", sync_after=True)  # noqa: SLF001

        print(f"MatcherTimer: instrumented {len(self._matchers)} matcher(s); discarding {self.warmup_steps} warmup steps.")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not self._originals:
            return
        for key in self._step_bucket:
            self._step_bucket[key] = 0.0
        self._sync()
        self._step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._originals or self._step_start is None:
            return
        self._sync()
        step_time = time.perf_counter() - self._step_start
        self._step_start = None
        self._seen_steps += 1

        if self._batch_size is None:
            self._batch_size = _infer_batch_size(batch)

        if self._seen_steps <= self.warmup_steps:
            return

        record = {"step": float(step_time), **{k: self._step_bucket[k] for k in ("total", *_MATCHER_BUCKETS)}}
        # dtoh is measured inside _prepare_costs, so uncrossing them keeps the buckets additive.
        record["prep"] = max(record["prep"] - record["dtoh"], 0.0)
        record["other"] = max(record["step"] - record["total"], 0.0)
        self._records.append(record)

        if self.print_every and len(self._records) % self.print_every == 0:
            n = len(self._records)
            share = 100.0 * np.median([r["total"] for r in self._records]) / np.median([r["step"] for r in self._records])
            print(f"MatcherTimer: {n} steps recorded, matcher = {share:.1f}% of step (median)")

    def on_train_end(self, trainer, pl_module) -> None:
        self._restore()
        if not self._records:
            print("MatcherTimer: no steps recorded past warmup; increase max_steps or lower warmup_steps.")
            return
        self._summary = self._summarise()
        self._write(trainer)
        self._report()

    def teardown(self, trainer, pl_module, stage) -> None:
        self._restore()

    def _restore(self) -> None:
        for matcher, saved in self._originals:
            for name, fn in saved.items():
                setattr(matcher, name, fn)
        self._originals = []

    # ------------------------------------------------------------------ reporting

    def _summarise(self) -> dict:
        keys = ("step", "total", "prep", "dtoh", "solve", "device", "other")
        arrays = {k: np.array([r[k] for r in self._records], dtype=np.float64) for k in keys}
        median_step = float(np.median(arrays["step"]))
        summary = {
            "num_steps": len(self._records),
            "warmup_steps_discarded": self.warmup_steps,
            "batch_size": self._batch_size,
            "median_step_s": median_step,
            "mean_step_s": float(arrays["step"].mean()),
            "samples_per_s": (self._batch_size / median_step) if self._batch_size else None,
            "device_fallbacks": sum(m.device_fallbacks for m in self._matchers),
            "buckets": {},
        }
        for key in ("total", "prep", "dtoh", "solve", "device", "other"):
            summary["buckets"][key] = {
                "median_s": float(np.median(arrays[key])),
                "mean_s": float(arrays[key].mean()),
                "std_s": float(arrays[key].std()),
                "median_pct_of_step": 100.0 * float(np.median(arrays[key])) / median_step,
            }
        return summary

    def _write(self, trainer) -> None:
        out_dir = Path(trainer.log_dir) if trainer.log_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"{self.output_name}.npz",
            **{k: np.array([r[k] for r in self._records], dtype=np.float64) for k in self._records[0]},
        )
        with (out_dir / f"{self.output_name}.json").open("w") as f:
            json.dump(self._summary, f, indent=2)
        self._out_dir = out_dir

    def _report(self) -> None:
        s = self._summary
        rows = [
            ("matcher, total", "total"),
            ("  device-side prep + length sync", "prep"),
            ("  device->host copy", "dtoh"),
            ("  host LAP solve", "solve"),
            ("  device solver", "device"),
            ("everything else", "other"),
        ]
        print("-" * 80)
        print(f"MatcherTimer: {s['num_steps']} steps after {s['warmup_steps_discarded']} warmup steps")
        print(f"  median step {1000 * s['median_step_s']:.1f} ms   mean {1000 * s['mean_step_s']:.1f} ms", end="")
        if s["samples_per_s"]:
            print(f"   ({s['samples_per_s']:.0f} samples/s, batch {s['batch_size']})")
        else:
            print()
        print(f"  {'bucket':<34}{'median (ms)':>14}{'% of step':>12}{'std (ms)':>12}")
        for label, key in rows:
            b = s["buckets"][key]
            if key == "device" and b["median_s"] == 0.0:
                continue
            print(f"  {label:<34}{1000 * b['median_s']:>14.1f}{b['median_pct_of_step']:>12.1f}{1000 * b['std_s']:>12.1f}")
        if s["device_fallbacks"]:
            print(f"  device solver fell back to the host on {s['device_fallbacks']} problems")
        print(f"  wrote {getattr(self, '_out_dir', '?')}/{self.output_name}.{{npz,json}}")
        print("-" * 80)


def _infer_batch_size(batch) -> int | None:
    """Best-effort sample count for one batch, for a samples/s figure."""
    candidates = batch
    if isinstance(batch, (tuple, list)) and batch:
        candidates = batch[0]
    if isinstance(candidates, dict):
        for value in candidates.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[0])
    elif isinstance(candidates, torch.Tensor) and candidates.ndim >= 1:
        return int(candidates.shape[0])
    return None
