"""``KerasMaskFormer.quant_losses()`` counts each keras layer once.

The task modules are registered twice in the torch module tree (``model.tasks`` and ``model.decoder.tasks``), so
``keras_layers()`` yields every task-head layer twice. ``quant_losses()`` used to sum over that walk: a Polaris run on
real CLIC events (beta0 1e-12) recorded 211 walked vs 183 unique layers and a penalty of 3.418 vs 3.137 (+8.9%).

On the org model with the production spec (beta0 1e-12, CLIC dummy events): 211 walked, 183 unique; ``quant_losses()``
equals the per-unique-layer sum (3.13701 here); the old walk (control) is larger by the task heads' terms, by the same
fraction as on real events: +8.945% here, +8.94% there.
"""

import copy

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import keras
import org_model_utils as U  # ty: ignore [unresolved-import]

from hepattn.keras import callbacks as C


def walked_sum(model) -> float:
    """The old quant_losses(): every layer of the keras_layers() walk, duplicates included."""
    return float(torch.stack([torch.as_tensor(t) for layer in model.keras_layers() for t in layer.losses]).sum())


def unique_sum(model) -> float:
    return float(torch.stack([torch.as_tensor(t) for layer in model.unique_keras_layers() for t in layer.losses]).sum())


@pytest.fixture(scope="module")
def generated() -> dict:
    return U.hybrid_config()


def test_generated_org_model_counts_each_layer_once(generated):
    cfg = copy.deepcopy(generated)
    cfg["model"]["model"]["init_args"]["quant"]["ebops"] = {"beta0": 1.0e-12}
    keras.utils.set_random_seed(0)  # the WRAP datalanes fit their integer bits to the activations: EBOPs depend on the init
    model = U.instantiate(cfg, quantized=True)
    inputs, _ = U.clic_dummy_batch(num_events=2)
    with torch.no_grad():
        model.materialize(inputs)
        model.train()(inputs)
    walked, unique = list(model.keras_layers()), model.unique_keras_layers()
    assert (len(walked), len(unique)) == (211, 183)
    task_layers = {id(layer) for task in model.tasks for layer in task.modules() if layer in walked}
    assert len(walked) - len(unique) == len(task_layers) > 0, "the duplicates are not exactly the task-head layers"

    new = float(model.quant_losses())
    assert new == pytest.approx(unique_sum(model), rel=1e-6)
    old = walked_sum(model)
    # measured on CLIC dummy events (offsets 0, 2, 4): new 3.13701 / 3.13733, old 3.41763 / 3.41794, +8.9445% to +8.9454%
    # (spread 1e-5; tolerance 2e-5). Real events: 3.418 / 3.137437 - 1 = +8.94%
    assert new == pytest.approx(3.1370, abs=7e-4)

    # EBOPs at init of the production spec, the input of the beta setpoints: every leaf's exact
    # (int64) EBOPs after one training forward of the seeded model: 250912214913 (unseeded builds read 2.50911e11 to
    # 2.50913e11: the WRAP datalanes fit their integer bits to the activations; 250913099895 after three training
    # forwards)
    total = C.total_ebops(model)
    assert total == pytest.approx(250912214913, rel=1e-6)
    assert (old - new) / new == pytest.approx(0.08945, abs=2e-5), (old, new)
