"""Checks on the Phase-0 attribution callback.

The callback is measurement infrastructure for a go/no-go gate, so the thing worth testing is
not that it produces plausible numbers but that its bookkeeping is sound: buckets that stay
additive, a matcher left exactly as it was found, and a graceful no-op when there is nothing
to time.
"""

import json

import numpy as np
import torch
from torch import nn

from hepattn.callbacks.matcher_timer import MatcherTimer
from hepattn.models.matcher import Matcher


class DummyTrainer:
    def __init__(self, log_dir):
        self.log_dir = str(log_dir)


class DummyModule(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher


def _run_steps(callback, trainer, module, num_steps, batch_size=3, num_pred=6, num_true=4):
    torch.manual_seed(0)
    callback.on_train_start(trainer, module)
    for _ in range(num_steps):
        batch = {"hit": torch.zeros(batch_size, 5)}
        callback.on_train_batch_start(trainer, module, batch, 0)
        costs = torch.rand(batch_size, num_pred, num_true)
        object_valid = torch.ones(batch_size, num_true, dtype=torch.bool)
        module.matcher(costs, object_valid, None)
        callback.on_train_batch_end(trainer, module, None, batch, 0)
    callback.on_train_end(trainer, module)


def test_buckets_are_additive_and_results_are_written(tmp_path):
    matcher = Matcher(default_solver="scipy", adaptive_solver=False)
    module = DummyModule(matcher)
    trainer = DummyTrainer(tmp_path)
    callback = MatcherTimer(warmup_steps=1, output_name="timing")

    _run_steps(callback, trainer, module, num_steps=4)

    summary = json.loads((tmp_path / "timing.json").read_text())
    assert summary["num_steps"] == 3
    assert summary["batch_size"] == 3

    with np.load(tmp_path / "timing.npz") as data:
        step = data["step"]
        parts = data["prep"] + data["dtoh"] + data["solve"] + data["device"] + data["other"]
        # The sub-buckets partition the step: the matcher's own total is prep+dtoh+solve, and
        # "other" is the remainder. Slack is the wrapper overhead, which is microseconds.
        assert np.all(parts <= step + 1e-3)
        assert np.all(data["total"] <= step + 1e-3)
        assert np.all(data["solve"] > 0)
        # No CUDA here, so nothing is staged through the pinned buffer.
        assert np.all(data["dtoh"] == 0)


def test_matcher_is_restored_and_still_matches(tmp_path):
    matcher = Matcher(default_solver="scipy", adaptive_solver=False)
    module = DummyModule(matcher)
    originals = {name: getattr(matcher, name) for name in ("forward", "_prepare_costs", "_stage_to_host", "_solve", "_match_on_device")}

    callback = MatcherTimer(warmup_steps=0, output_name="timing")
    _run_steps(callback, DummyTrainer(tmp_path), module, num_steps=2)

    for name, fn in originals.items():
        assert getattr(matcher, name) == fn

    costs = torch.rand(2, 5, 3)
    valid = torch.ones(2, 3, dtype=torch.bool)
    pred_idxs = matcher(costs, valid, None)
    assert pred_idxs.shape == (2, 5)
    assert sorted(pred_idxs[0].tolist()) == list(range(5))


def test_no_matcher_is_a_no_op(tmp_path):
    module = nn.Linear(2, 2)
    callback = MatcherTimer(warmup_steps=0, output_name="timing")

    callback.on_train_start(DummyTrainer(tmp_path), module)
    callback.on_train_batch_start(DummyTrainer(tmp_path), module, {"hit": torch.zeros(2, 2)}, 0)
    callback.on_train_batch_end(DummyTrainer(tmp_path), module, None, {"hit": torch.zeros(2, 2)}, 0)
    callback.on_train_end(DummyTrainer(tmp_path), module)

    assert not list(tmp_path.glob("*.npz"))
