from types import SimpleNamespace

import pytest
from torch import nn

from hepattn.models.wrapper import ModelWrapper

LRS = {"initial": 1e-6, "max": 5e-4, "end": 1e-6, "pct_start": 0.05, "weight_decay": 0.1}


def scheduled_betas(optimizer, lrs_config, steps=50):
    wrapper = ModelWrapper(name="x", model=nn.Linear(4, 4), lrs_config=lrs_config, optimizer=optimizer)
    wrapper._trainer = SimpleNamespace(estimated_stepping_batches=100)  # noqa: SLF001
    [opt], [sch] = wrapper.configure_optimizers()
    betas = []
    for _ in range(steps):
        opt.step()
        sch["scheduler"].step()
        betas.append(opt.param_groups[0]["betas"][0])
    return betas, opt.param_groups[0]["lr"]


@pytest.mark.parametrize("optimizer", ["Lion", "AdamW"])
def test_cycle_momentum(optimizer):
    # the default keeps OneCycleLR's beta1 cycling, as before
    betas, lr = scheduled_betas(optimizer, LRS)
    assert len(set(betas)) > 1
    # cycle_momentum false leaves beta1 alone while the LR still follows the one-cycle schedule
    betas_off, lr_off = scheduled_betas(optimizer, {**LRS, "cycle_momentum": False})
    assert len(set(betas_off)) == 1
    assert lr_off == pytest.approx(lr)
