"""The org/main CLIC model (base.yaml + linformer.yaml) as torch and hybrid twins, built the way training builds them.

Both models are instantiated by jsonargparse from a YAML ``model:`` section, as the LightningCLI does in ``main.py``,
so a config key that the hybrid signature rejects or misreads fails here too.

- The torch reference comes straight from the org configs (``merge_org_configs``).
- ``hybrid_config`` swaps the network for ``KerasMaskFormer`` and adds the HGQ2 production spec; nothing else changes.
- ``to_dense_attention`` swaps the masked-Linformer attention for dense attention on both sides (every other part
  unchanged: PR #8 layers, SwiGLU, phi sorting, tasks, matcher, losses).
"""

import copy
import pathlib

import torch
import yaml
from jsonargparse import ArgumentParser

import hepattn
from hepattn.experiments.clic.pflow_data import CLICDataset

CLIC = pathlib.Path(hepattn.__file__).parent / "experiments/clic"
CONFIGS = CLIC / "configs"
SCALE_DICT = str(CONFIGS / "clic_var_transform.yaml")
NUM_QUERIES = 150
MAX_NODES = 160
LINFORMER_KEYS = ("linformer_seq_len", "linformer_k")
# HGQ2 production quantization spec (weights 8 bits with 2 integer bits; datalanes 4 + 8; tables 2 + 10).
# q_type/place are selectors in HGQ2 scopes; the type is chosen through default_q_type.
PROD_QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 0.0},
}


def merge(base: dict, overlay: dict) -> dict:
    """Overlay rule: dicts merge key by key, other values replace (the effective LightningCLI result for these configs)."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def merge_org_configs() -> dict:
    """org/main's linformer model config: base.yaml with linformer.yaml over it."""
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    return merge(base, yaml.safe_load((CONFIGS / "linformer.yaml").read_text()))


def hybrid_config(cfg: dict | None = None) -> dict:
    """``cfg`` (default: ``merge_org_configs()``) with the network swapped for ``KerasMaskFormer`` and the HGQ2 spec.

    ``KerasMaskFormer`` takes the encoder as a plain dict (it builds the layers inside the HGQ2 config scopes) and
    passes the model dim itself. The matcher solver becomes scipy: ``lap1015`` needs the compiled extension.
    """
    cfg = copy.deepcopy(merge_org_configs() if cfg is None else cfg)
    net_cls = cfg["model"]["model"]
    net = net_cls["init_args"]
    net_cls["class_path"] = "hepattn.keras.maskformer.KerasMaskFormer"
    encoder = dict(net["encoder"]["init_args"])
    assert encoder.pop("dim") == net["dim"], "encoder dim differs from the model dim"
    net["encoder"] = encoder
    net["matcher"]["init_args"]["default_solver"] = "scipy"
    net["quant"] = copy.deepcopy(PROD_QUANT)
    return cfg


def _dense(attn_kwargs: dict) -> None:
    for key in LINFORMER_KEYS:
        attn_kwargs.pop(key)
    attn_kwargs["attn_type"] = "torch"


def to_dense_attention(cfg: dict) -> dict:
    """The same model section with dense attention everywhere (``cfg`` is a full config; a copy is returned)."""
    cfg = copy.deepcopy(cfg)
    net = cfg["model"]["model"]["init_args"]
    encoder = net["encoder"]["init_args"] if "class_path" in net["encoder"] else net["encoder"]
    _dense(encoder["attn_kwargs"])
    encoder["attn_type"] = "torch"
    _dense(net["decoder"]["decoder_layer_config"]["attn_kwargs"])
    return cfg


def torch_reference_config(cfg: dict) -> dict:
    """Org config ready to run on CPU: the encoder's flash-varlen becomes torch (same maths), the solver scipy."""
    cfg = copy.deepcopy(cfg)
    net = cfg["model"]["model"]["init_args"]
    if net["encoder"]["init_args"]["attn_type"] in {"flash", "flash-varlen"}:
        net["encoder"]["init_args"]["attn_type"] = "torch"
    net["matcher"]["init_args"].update({"default_solver": "scipy", "parallel_solver": False})
    return cfg


def instantiate(cfg: dict, quantized: bool | None = None):
    """The Lightning module of a full config, built by jsonargparse. Returns its ``.model``, in eval mode.

    ``quantized=False`` drops the HGQ2 ``quant`` section of a KerasMaskFormer config (the float twin).
    """
    from hepattn.experiments.clic.lightning_module import MPflow  # noqa: PLC0415

    cfg = copy.deepcopy(cfg)
    net = cfg["model"]["model"]["init_args"]
    if quantized is False and cfg["model"]["model"]["class_path"].endswith("KerasMaskFormer"):
        net.pop("quant", None)
    for task in net["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task.get("init_args", {}):
            task["init_args"]["scale_dict_path"] = SCALE_DICT
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(MPflow, "model", fail_untyped=False)
    init = parser.instantiate_classes(parser.parse_object({"model": {**cfg["model"], "name": cfg["name"]}}))
    return init.model.model.eval()


def clic_dummy_batch(num_events: int = 2, seed: int = 0) -> tuple[dict, dict]:
    """CLIC dummy events (every event has at least one padded node: n_nodes < max_nodes)."""
    ds = CLICDataset(
        filepath="dummy",
        inputs={"node": ["features"]},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=SCALE_DICT,
        num_events=num_events,
        num_objects=NUM_QUERIES,
        max_nodes=MAX_NODES,
        dummy_data=True,
    )
    events = [ds[seed + i] for i in range(num_events)]
    inputs = {k: torch.stack([e[0][k] for e in events]) for k in events[0][0]}
    targets = {k: torch.stack([e[1][k] for e in events]) for k in events[0][1]}
    return inputs, targets
