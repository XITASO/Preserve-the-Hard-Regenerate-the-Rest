from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from guided_generation.config import ConfigError, config_to_env, load_pipeline_config
from guided_generation.main_scripts.run_pipeline import (
    apply_smoke_overrides,
    build_commands,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "dataset", "tau", "real_split", "sample_count"),
    [
        ("cityscapes.yaml", "cityscapes", 0.10, 0.10, 297),
        ("bdd100k.yaml", "bdd100k", 0.10, 0.10, 700),
        ("uavid.yaml", "uavid", 0.15, 1.0, 200),
    ],
)
def test_paper_configs(
    filename: str,
    dataset: str,
    tau: float,
    real_split: float,
    sample_count: int,
) -> None:
    config = load_pipeline_config(REPO_ROOT / "configs" / filename)
    assert config["dataset"]["name"] == dataset
    assert config["selection"]["tau"] == tau
    assert config["training"]["real_split"] == real_split
    assert config["selection"]["num_samples"] == sample_count
    assert config["generation"]["max_samples"] == sample_count
    assert config["generation"]["inpainter"] == "sdxl_inpainting"
    assert config["model"]["encoder_name"] == "vit_small_patch14_dinov2"


def test_smoke_overrides_only_runtime_limits() -> None:
    config = load_pipeline_config(REPO_ROOT / "configs" / "cityscapes.yaml")
    original = deepcopy(config)
    apply_smoke_overrides(config)
    assert config["selection"]["num_samples"] == 2
    assert config["selection"]["max_samples"] == 2
    assert config["generation"]["num_steps"] == 2
    assert config["training"]["fine_tune_seeds"] == [42]
    assert config["training"]["batch_size"] == 1
    assert config["training"]["num_workers"] == 0
    assert config["model"] == original["model"]
    assert config["selection"]["tau"] == original["selection"]["tau"]
    commands = build_commands(config, "model.ckpt")
    assert "--batch_size" in commands[1][0]
    assert "--num_workers" in commands[1][0]
    assert "--batch_size" in commands[4][0]
    assert "--num_workers" in commands[4][0]


def test_backbone_and_inpainter_are_config_only_changes() -> None:
    config = load_pipeline_config(REPO_ROOT / "configs" / "cityscapes.yaml")
    config["model"]["encoder_name"] = "resnet50.a1_in1k"
    config["generation"]["inpainter"] = "flux_fill"
    config["generation"]["model_id"] = "black-forest-labs/FLUX.1-Fill-dev"
    commands = build_commands(config, "model.ckpt")
    for stage in (1, 2, 4):
        assert "resnet50.a1_in1k" in commands[stage][0]
    assert "flux_fill" in commands[3][0]
    assert "black-forest-labs/FLUX.1-Fill-dev" in commands[3][0]


def test_shell_environment_maps_shared_paths() -> None:
    config = load_pipeline_config(REPO_ROOT / "configs" / "bdd100k.yaml")
    environment = config_to_env(config)
    assert environment["REAL_ROOT"] == environment["DATA_ROOT"]
    assert environment["CACHE_DIR"] == environment["ROOT_DIR"]
    assert environment["OUTPUT_FOLDER"] == environment["SYN_ROOT"]
    assert environment["NUM_RUNS"] == "5"


def test_rejects_wrong_registered_ignore_index(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "uavid.yaml").read_text(encoding="utf-8")
    )
    source["dataset"]["ignore_index"] = 255
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="ignore_index"):
        load_pipeline_config(path)


def test_rejects_unknown_inpainter(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "cityscapes.yaml").read_text(encoding="utf-8")
    )
    source["generation"]["inpainter"] = "not-a-model"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported generation.inpainter"):
        load_pipeline_config(path)


def test_rejects_one_step_native_diffusers_inpainting(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "cityscapes.yaml").read_text(encoding="utf-8")
    )
    source["generation"]["inpainter"] = "sdxl_diffusers_inpainting"
    source["generation"]["num_steps"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="num_steps >= 2"):
        load_pipeline_config(path)
