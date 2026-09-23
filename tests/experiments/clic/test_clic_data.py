from pathlib import Path

import pytest
import torch
import yaml

from hepattn.experiments.clic.pflow_data import CLICDataset

torch.manual_seed(42)


class TestCLICDataset:
    @pytest.fixture
    def clic_dataset(self):
        config_path = Path("src/hepattn/experiments/clic/configs/base.yaml")
        config = yaml.safe_load(config_path.read_text())["data"]

        filepath = config["valid_path"]
        if not Path(filepath).is_file():
            pytest.skip(f"CLIC validation file not available: {filepath}")

        return CLICDataset(
            filepath=filepath,
            inputs=config["inputs"],
            targets=config["targets"],
            scale_dict_path="src/hepattn/experiments/clic/configs/clic_var_transform.yaml",
            num_events=20,
            num_objects=config["num_objects"],
            max_nodes=config["max_nodes"],
        )

    @pytest.mark.requiresdata
    def test_phi_channels_are_sine_and_cosine(self, clic_dataset):
        """The cosphi/sinphi input channels must hold cos(phi)/sin(phi), for topoclusters as well as tracks.

        They were once filled with phi itself for topoclusters, which is not caught by any shape
        or dtype check: the channels stay finite and the model trains, it just never sees the
        azimuth as a continuous pair.
        """
        # Channel order follows the node_features dict: pt, eta, phi, cosphi, sinphi, ...
        phi_idx, cosphi_idx, sinphi_idx = 2, 3, 4

        for i in range(10):
            inputs, _ = clic_dataset[i]
            valid = inputs["node_valid"].bool()
            features = inputs["node_features"][valid]
            phi = features[:, phi_idx]
            cosphi = features[:, cosphi_idx]
            sinphi = features[:, sinphi_idx]

            torch.testing.assert_close(cosphi, torch.cos(phi), atol=1e-5, rtol=0)
            torch.testing.assert_close(sinphi, torch.sin(phi), atol=1e-5, rtol=0)
            torch.testing.assert_close(cosphi**2 + sinphi**2, torch.ones_like(phi), atol=1e-5, rtol=0)
