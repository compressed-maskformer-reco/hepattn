"""Checks on the Phase-1 cost-dump callback.

The callback's whole purpose is to hand a file to ``bench_device_matcher.py --costs``, so what
is worth testing is the contract at that seam: the keys the replay reads, tensors that still
match what the matcher was handed, a matcher left exactly as it was found, and no interference
with the matching itself.
"""

import json

import pytest
import torch
from torch import nn

from hepattn.callbacks.matcher_cost_dump import MatcherCostDump
from hepattn.models.matcher import Matcher


class DummyTrainer:
    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self.should_stop = False


class DummyModule(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher


def _run_steps(callback, trainer, module, num_steps, batch_size=6, num_pred=5, num_true=4, seed=0):
    """Drive the lightning hooks by hand, returning the costs passed on each step."""
    torch.manual_seed(seed)
    callback.on_train_start(trainer, module)
    seen = []
    for _ in range(num_steps):
        callback.on_train_batch_start(trainer, module, {}, 0)
        costs = torch.rand(batch_size, num_pred, num_true)
        object_valid = torch.ones(batch_size, num_true, dtype=torch.bool)
        object_valid[0, -1] = False
        seen.append(costs)
        module.matcher(costs, object_valid, None)
        callback.on_train_batch_end(trainer, module, None, {}, 0)
    callback.on_train_end(trainer, module)
    return seen


@pytest.fixture
def module():
    return DummyModule(Matcher(default_solver="scipy", adaptive_solver=False))


def test_dumps_the_chosen_step(tmp_path, module):
    """The file holds the costs from ``dump_step``, not from whichever step ran first."""
    callback = MatcherCostDump(dump_step=2, output_name="costs", stop_after=False)
    trainer = DummyTrainer(tmp_path)
    seen = _run_steps(callback, trainer, module, num_steps=4)

    blob = torch.load(tmp_path / "costs.pt", weights_only=True)
    assert torch.equal(blob["costs"], seen[2])
    assert not any(torch.equal(blob["costs"], seen[i]) for i in (0, 1, 3))


def test_schema_matches_the_replay_reader(tmp_path, module):
    """The keys and layouts ``bench_device_matcher.py --costs`` expects."""
    callback = MatcherCostDump(dump_step=0, output_name="costs", stop_after=False)
    trainer = DummyTrainer(tmp_path)
    _run_steps(callback, trainer, module, num_steps=1, batch_size=6, num_pred=5, num_true=4)

    blob = torch.load(tmp_path / "costs.pt", weights_only=True)
    assert set(blob) == {"costs", "object_valid_mask"}  # no query mask was passed
    assert blob["costs"].shape == (6, 5, 4)
    assert blob["object_valid_mask"].shape == (6, 4)
    assert blob["object_valid_mask"].dtype == torch.bool
    assert not blob["costs"].is_cuda

    manifest = json.loads((tmp_path / "costs.json").read_text())
    assert manifest["shape"] == [6, 5, 4]
    assert manifest["targets_min"] == 3  # one event had a target masked off
    assert manifest["targets_max"] == 4
    assert manifest["non_finite"] == 0


def test_subsamples_by_striding(tmp_path, module):
    """``max_problems`` strides, so a layer-major stacked batch stays evenly represented."""
    callback = MatcherCostDump(dump_step=0, output_name="costs", max_problems=3, stop_after=False)
    trainer = DummyTrainer(tmp_path)
    seen = _run_steps(callback, trainer, module, num_steps=1, batch_size=6)

    blob = torch.load(tmp_path / "costs.pt", weights_only=True)
    assert blob["costs"].shape[0] == 3
    assert torch.equal(blob["costs"], seen[0][0:6:2])


def test_stops_training_once_written(tmp_path, module):
    callback = MatcherCostDump(dump_step=0, output_name="costs", stop_after=True)
    trainer = DummyTrainer(tmp_path)
    _run_steps(callback, trainer, module, num_steps=1)
    assert trainer.should_stop


def test_restores_the_matcher_and_leaves_matching_unchanged(tmp_path, module):
    """The callback is measurement scaffolding: it must not survive the run or change answers."""
    original = module.matcher.forward
    costs = torch.rand(6, 5, 4)
    object_valid = torch.ones(6, 4, dtype=torch.bool)
    expected = module.matcher(costs, object_valid, None)

    callback = MatcherCostDump(dump_step=0, output_name="costs", stop_after=False)
    trainer = DummyTrainer(tmp_path)
    callback.on_train_start(trainer, module)
    callback.on_train_batch_start(trainer, module, {}, 0)
    during = module.matcher(costs, object_valid, None)
    callback.on_train_batch_end(trainer, module, None, {}, 0)
    callback.on_train_end(trainer, module)

    assert torch.equal(during, expected)
    assert module.matcher.forward == original
    assert torch.equal(module.matcher(costs, object_valid, None), expected)


def test_no_matcher_is_a_no_op(tmp_path):
    """A model without a matcher must not crash the run it was attached to."""
    callback = MatcherCostDump(dump_step=0, stop_after=False)
    trainer = DummyTrainer(tmp_path)
    plain = nn.Linear(2, 2)

    callback.on_train_start(trainer, plain)
    callback.on_train_batch_start(trainer, plain, {}, 0)
    callback.on_train_batch_end(trainer, plain, None, {}, 0)
    callback.on_train_end(trainer, plain)

    assert callback.written_path is None
    assert not list(tmp_path.glob("*.pt"))


def test_ending_before_the_dump_step_writes_nothing(tmp_path, module):
    callback = MatcherCostDump(dump_step=10, output_name="costs", stop_after=False)
    trainer = DummyTrainer(tmp_path)
    _run_steps(callback, trainer, module, num_steps=3)

    assert callback.written_path is None
    assert not list(tmp_path.glob("*.pt"))
