import copy
import math
from pathlib import Path

import pytest
import torch
import yaml
from jsonargparse import ActionConfigFile, ArgumentParser
from torch.utils.data import DataLoader

from hepattn.experiments.clic import main
from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.experiments.clic.pflow_data import CLICDataset
from hepattn.models.encoder import PatchEncoderLayer

from ..utils import run_test  # noqa: TID252

CONFIG_DIR = Path(main.__file__).parent / "configs"


def test_clic() -> None:
    run_test(main, "tests/experiments/clic/test_clic.yaml")


def test_clic_patch() -> None:
    run_test(main, "tests/experiments/clic/test_clic_patch.yaml")


def overlay_on_base(overlay: str):
    # Compose base.yaml and an overlay the way `main.py fit --config configs/<overlay>` does
    # (jsonargparse merges nested init_args); the caller must chdir to the experiment dir first.
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed_everything", type=int)
    parser.add_argument("--data", type=dict)
    parser.add_argument("--trainer", type=dict)
    parser.add_class_arguments(MPflow, "model")
    # main.py links --name to --model.name; this parser has no link, so pass it
    cfg = parser.parse_args(["--config", str(CONFIG_DIR / "base.yaml"), "--config", str(CONFIG_DIR / overlay), "--model.name", "x"])
    return cfg, parser.instantiate_classes(cfg).model.model


def dummy_batch(cfg, num_events=2):
    data = CLICDataset(
        filepath="dummy",
        inputs=cfg.data["inputs"],
        targets=cfg.data["targets"],
        scale_dict_path=cfg.data["scale_dict_path"],
        num_events=num_events,
        dummy_data=True,
    )
    return next(iter(DataLoader(data, batch_size=num_events)))


def total_loss(model, outputs, targets):
    _, _, losses = model.loss(outputs, targets)
    return sum(loss for layer in losses.values() for task in layer.values() for loss in task.values())


def test_patch_overlay_on_base(monkeypatch) -> None:
    # The patch overlay on base.yaml, then the full-size model on a dummy CLIC batch
    monkeypatch.chdir(CONFIG_DIR.parent)  # base.yaml's scale_dict_path is relative to the experiment dir
    cfg, model = overlay_on_base("patch.yaml")
    assert cfg.name == "clic_patch_p16_phicalo"
    assert model.input_sort_field == "phi_calo"
    assert model.input_sort_period == pytest.approx(2 * math.pi)
    encoder = model.encoder
    assert encoder.patch_size == 16
    assert encoder.patch_global
    assert encoder.attn_type == "patch"
    assert all(layer.attn.fn.attn_type == "torch" for layer in encoder.layers)
    # the rest of base.yaml's encoder is kept
    assert len(encoder.layers) == 6
    assert all(isinstance(layer, PatchEncoderLayer) for layer in encoder.layers)
    assert encoder.num_register_tokens == 8
    assert encoder.value_residual
    assert encoder.layers[0].attn.fn.num_heads == 16
    # and the decoder is base.yaml's quadratic one
    assert len(model.decoder.decoder_layers) == 4
    assert all(layer.attn_type == "torch" for layer in model.decoder.decoder_layers)

    inputs, targets = dummy_batch(cfg)
    torch.manual_seed(0)
    outputs = model(inputs)
    num_nodes = inputs["node_valid"].shape[-1]
    assert outputs["final"]["mask"]["pflow_node_logit"].shape == (2, 150, num_nodes)
    assert outputs["final"]["incidence"]["pflow_incidence"].shape == (2, 150, num_nodes)

    total = total_loss(model, outputs, targets)
    assert total.isfinite()
    total.backward()
    no_grad = [n for n, p in encoder.named_parameters() if p.grad is None or not p.grad.isfinite().all()]
    assert not no_grad, no_grad


# Every float node input of the CLIC dataset
PADDED_INPUT_KEYS = ("node_features", "node_eta", "node_phi", "node_phi_calo", "node_eta_calo", "node_e", "node_pt", "node_sinphi", "node_cosphi", "node_is_track")
# The raw node fields IncidenceBasedRegressionTask.get_proxy_feats reads (its `fields` and `is_track`), see below
REGRESSION_RAW_KEYS = ("node_e", "node_pt", "node_eta", "node_sinphi", "node_cosphi", "node_is_track")
# Shared, not patch-specific (traced in patch-attention-env/analysis/phat16_review_20261019/02_padding_nonfinite.md):
# IncidenceBasedRegressionTask.get_proxy_feats (models/task.py) contracts the raw node fields of all node slots,
# padding included, with the predicted incidence (torch.bmm / einsum). A zero incidence weight times a padded inf/NaN
# is NaN, so the regression outputs and the loss become NaN. base.yaml fails the same way. The dataset pads with zeros.
SHARED_REGRESSION_LEAK = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="shared IncidenceBasedRegressionTask.get_proxy_feats: 0-weight x padded inf/NaN raw node field = NaN regression and loss",
)


def patch_model_with_garbage(fill, keys, overlay="patch.yaml"):
    cfg, model = overlay_on_base(overlay)
    model.eval()
    inputs, targets = dummy_batch(cfg)
    valid = inputs["node_valid"]
    garbage = {k: (v.masked_fill(~valid if v.ndim == 2 else ~valid[..., None], fill) if k in keys else v) for k, v in inputs.items()}
    with torch.no_grad():
        out, out_g = model(inputs), model(garbage)
        # model.loss reorders the output dict in place by the matching, which inf/NaN regression costs change,
        # so compute the losses on copies and compare the outputs in query order
        loss, loss_g = total_loss(model, copy.deepcopy(out), targets), total_loss(model, copy.deepcopy(out_g), targets)
    return valid, out, out_g, loss, loss_g


@pytest.mark.parametrize("fill", [1e4, float("inf"), float("nan")])
def test_patch_model_ignores_padded_nodes(monkeypatch, fill) -> None:
    # Whatever every padded node input holds, the full patch model's mask, incidence and class outputs for the real
    # nodes are unchanged (the encoder zeroes padded slots, so nothing reaches the decoder)
    monkeypatch.chdir(CONFIG_DIR.parent)
    valid, out, out_g, _, _ = patch_model_with_garbage(fill, PADDED_INPUT_KEYS)
    for task, key in (("mask", "pflow_node_logit"), ("incidence", "pflow_incidence")):
        ref, got = out["final"][task][key], out_g["final"][task][key]
        torch.testing.assert_close(got.transpose(1, 2)[valid], ref.transpose(1, 2)[valid])
    torch.testing.assert_close(out_g["final"]["classification"]["pflow_logit"], out["final"]["classification"]["pflow_logit"])


@pytest.mark.parametrize("fill", [1e4, float("inf"), float("nan")])
def test_patch_model_loss_ignores_padded_non_regression_inputs(monkeypatch, fill) -> None:
    # Padded garbage in every input the regression task does not read leaves all outputs and the loss unchanged:
    # the patch encoder has no padding leak of its own (base.yaml's exact-attention encoder does, for NaN/inf)
    monkeypatch.chdir(CONFIG_DIR.parent)
    keys = tuple(k for k in PADDED_INPUT_KEYS if k not in REGRESSION_RAW_KEYS)
    _, out, out_g, loss, loss_g = patch_model_with_garbage(fill, keys)
    torch.testing.assert_close(out_g["final"]["regression"]["pflow_regr"], out["final"]["regression"]["pflow_regr"])
    assert loss_g.isfinite()
    torch.testing.assert_close(loss_g, loss)


@pytest.mark.parametrize("fill", [1e4, pytest.param(float("inf"), marks=SHARED_REGRESSION_LEAK), pytest.param(float("nan"), marks=SHARED_REGRESSION_LEAK)])
def test_patch_model_loss_ignores_padded_nodes(monkeypatch, fill) -> None:
    # With every padded input filled, the regression outputs and the loss are unchanged and finite: true for a large
    # finite fill; inf/NaN fail in the shared regression task (SHARED_REGRESSION_LEAK), not in the patch encoder
    monkeypatch.chdir(CONFIG_DIR.parent)
    _, out, out_g, loss, loss_g = patch_model_with_garbage(fill, PADDED_INPUT_KEYS)
    torch.testing.assert_close(out_g["final"]["regression"]["pflow_regr"], out["final"]["regression"]["pflow_regr"])
    assert loss_g.isfinite()
    torch.testing.assert_close(loss_g, loss)


def test_patch_gmp_overlay_on_base(monkeypatch) -> None:
    # patch_gmp.yaml is patch.yaml plus GMP in every encoder layer, fed the calorimeter coordinates node_eta_calo /
    # node_phi_calo (tracks at the calorimeter front face, clusters at their barycentre); nothing else changes
    monkeypatch.chdir(CONFIG_DIR.parent)
    cfg, model = overlay_on_base("patch_gmp.yaml")
    _, patch = overlay_on_base("patch.yaml")
    assert cfg.name == "clic_patch_p16_phicalo_gmp"
    assert model.input_coord_fields == ["eta_calo", "phi_calo"]
    assert (model.input_sort_field, model.input_sort_period) == (patch.input_sort_field, patch.input_sort_period)
    assert patch.input_coord_fields is None
    assert patch.encoder.gmp is None
    assert all(layer.gmp is None for layer in patch.encoder.layers)
    encoder = model.encoder
    assert encoder.gmp == {"kernel_size": 3, "eta_cell": 0.2, "phi_cells": 32, "eta_max": 5.2}
    assert len(encoder.layers) == 6
    assert all(isinstance(layer, PatchEncoderLayer) and layer.gmp.geometry == encoder.gmp for layer in encoder.layers)
    # the same model as patch.yaml's plus 6 x 68,864 GMP parameters
    params = {n: p.shape for n, p in model.named_parameters()}
    patch_params = {n: p.shape for n, p in patch.named_parameters()}
    assert {n: s for n, s in params.items() if ".gmp." not in n} == patch_params
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in patch.parameters()) == 6 * 68_864

    inputs, targets = dummy_batch(cfg)
    torch.manual_seed(0)
    outputs = model(inputs)
    num_nodes = inputs["node_valid"].shape[-1]
    assert outputs["final"]["mask"]["pflow_node_logit"].shape == (2, 150, num_nodes)
    total = total_loss(model, outputs, targets)
    assert total.isfinite()
    total.backward()
    no_grad = [n for n, p in encoder.named_parameters() if p.grad is None or not p.grad.isfinite().all()]
    assert not no_grad, no_grad
    assert all(layer.gmp.dw.weight.grad.abs().sum() > 0 for layer in encoder.layers)


@pytest.mark.parametrize("fill", [1e4, float("inf"), float("nan")])
def test_patch_gmp_model_ignores_padded_nodes(monkeypatch, fill) -> None:
    # As test_patch_model_ignores_padded_nodes, with GMP: padded node_eta_calo / node_phi_calo (the GMP coordinates) included
    monkeypatch.chdir(CONFIG_DIR.parent)
    valid, out, out_g, _, _ = patch_model_with_garbage(fill, PADDED_INPUT_KEYS, overlay="patch_gmp.yaml")
    for task, key in (("mask", "pflow_node_logit"), ("incidence", "pflow_incidence")):
        ref, got = out["final"][task][key], out_g["final"][task][key]
        torch.testing.assert_close(got.transpose(1, 2)[valid], ref.transpose(1, 2)[valid])
    torch.testing.assert_close(out_g["final"]["classification"]["pflow_logit"], out["final"]["classification"]["pflow_logit"])


CLIC_VAL = Path("/eagle/hgcal-maskformer-fpga/clic_data/val_clic_fix.root")


@pytest.mark.requiresdata
@pytest.mark.skipif(not CLIC_VAL.exists(), reason="needs the CLIC validation file")
def test_gmp_coordinates_are_calorimeter_coordinates(monkeypatch) -> None:
    # node_eta_calo / node_phi_calo, the GMP coordinates: a track's (eta_int, phi_int) at the calorimeter front face, a
    # topo cluster's (eta, phi), raw and zero-padded. Checked against the dataset's own 27 features: columns 1 (scaled
    # eta), 5 (scaled eta_int) and 6 (phi_int); the features themselves are unchanged (27 of them).
    monkeypatch.chdir(CONFIG_DIR.parent)
    cfg, _ = overlay_on_base("patch_gmp.yaml")
    data = CLICDataset(filepath=str(CLIC_VAL), inputs=cfg.data["inputs"], targets=cfg.data["targets"], scale_dict_path=cfg.data["scale_dict_path"], num_events=20)
    eta_scaler = data.scaler.transforms["eta"]
    for idx in range(len(data)):
        inputs, _ = data[idx]
        features, valid, is_track = inputs["node_features"], inputs["node_valid"], inputs["node_is_track"].bool()
        assert features.shape[-1] == 27
        eta_calo, phi_calo = inputs["node_eta_calo"], inputs["node_phi_calo"]
        tracks, clusters = valid & is_track, valid & ~is_track
        torch.testing.assert_close(eta_scaler.transform(eta_calo[tracks]), features[tracks, 5])
        torch.testing.assert_close(phi_calo[tracks], features[tracks, 6])
        torch.testing.assert_close(eta_calo[clusters], inputs["node_eta"][clusters])
        torch.testing.assert_close(phi_calo[clusters], inputs["node_phi"][clusters])
        torch.testing.assert_close(eta_scaler.transform(eta_calo[clusters]), features[clusters, 1])
        assert (eta_calo[~valid] == 0).all()


@pytest.mark.parametrize(("recipe", "arch"), [("patch_gmp_lion400.yaml", "patch_gmp.yaml"), ("patch_lion400.yaml", "patch.yaml")])
def test_lion400_recipes(monkeypatch, recipe, arch) -> None:
    # The 400-epoch configs: the architecture of their overlay exactly, with the approved Lion recipe
    monkeypatch.chdir(CONFIG_DIR.parent)
    cfg, model = overlay_on_base(recipe)
    arch_cfg, arch_model = overlay_on_base(arch)
    assert cfg.model.model == arch_cfg.model.model
    assert {n: p.shape for n, p in model.named_parameters()} == {n: p.shape for n, p in arch_model.named_parameters()}
    assert cfg.model.optimizer == "Lion"
    assert cfg.model.lrs_config == {"initial": 1e-6, "max": 6.2e-4, "end": 1e-6, "pct_start": 0.05, "weight_decay": 0.1, "skip_scheduler": False, "cycle_momentum": False}
    assert cfg.trainer["max_epochs"] == 400
    assert cfg.trainer["gradient_clip_val"] == 0.1
    assert cfg.data["batch_size"] == 512


def test_clic_patch_gmp(tmp_path) -> None:
    # The CLI fit + test of test_clic_patch.yaml with the GMP of patch_gmp.yaml: the config round trip (the saved
    # config.yaml rebuilds the GMP model for the test stage) and a checkpoint holding the GMP weights
    config = yaml.safe_load(Path("tests/experiments/clic/test_clic_patch.yaml").read_text())
    config["name"] = "clic_patch_gmp_test"
    init_args = config["model"]["model"]["init_args"]
    init_args["input_coord_fields"] = ["eta_calo", "phi_calo"]
    init_args["encoder"]["init_args"]["gmp"] = {"kernel_size": 3, "eta_cell": 0.2, "phi_cells": 32, "eta_max": 5.2}
    path = tmp_path / "test_clic_patch_gmp.yaml"
    path.write_text(yaml.safe_dump(config))
    run_test(main, str(path))
    run_dir = max(Path("test_logs").glob("clic_patch_gmp_test_*"), key=lambda d: d.stat().st_mtime)
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())["model"]["model"]["init_args"]
    assert saved["input_coord_fields"] == ["eta_calo", "phi_calo"]
    assert saved["encoder"]["init_args"]["gmp"]["kernel_size"] == 3
    ckpt = torch.load(next((run_dir / "ckpts").glob("*.ckpt")), map_location="cpu", weights_only=False)
    assert "model.encoder.layers.0.gmp.dw.weight" in ckpt["state_dict"]


def test_linformer_encoder_overlay_keeps_decoder(monkeypatch) -> None:
    # The encoder-only linformer control: linformer in the encoder, base.yaml's exact decoder, patch.yaml's order
    monkeypatch.chdir(CONFIG_DIR.parent)
    cfg, model = overlay_on_base("linformer_encoder.yaml")
    _, patch = overlay_on_base("patch.yaml")
    assert model.encoder.attn_type == "linformer"
    assert all(layer.attn_type == "torch" for layer in model.decoder.decoder_layers)
    assert (model.input_sort_field, model.input_sort_period) == (patch.input_sort_field, patch.input_sort_period)
    # everything outside the encoder is the same model as the patch one
    outside = {n: p.shape for n, p in model.named_parameters() if not n.startswith("encoder.")}
    assert outside == {n: p.shape for n, p in patch.named_parameters() if not n.startswith("encoder.")}
    inputs, targets = dummy_batch(cfg)
    assert total_loss(model, model(inputs), targets).isfinite()
