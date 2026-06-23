from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml

from dataset_configs import get_dataset_spec


SUPPORTED_INPAINTERS = {
    "sdxl_inpainting",
    "sdxl_controlnet_inpainting",
    "sdxl_diffusers_inpainting",
    "sd15_diffusers_inpainting",
    "flux_fill",
}
SUPPORTED_SELECTORS = {
    "highest_entropy_class",
    "highest_entropy_class_multi",
    "lowest_entropy_class_multi",
    "random_class_multi",
    "random_square_region",
}


class ConfigError(ValueError):
    pass


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid '{name}' section.")
    return value


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    value = section.get(key)
    if value is None or value == "":
        raise ConfigError(f"Missing required setting '{section_name}.{key}'.")
    return value


def _fraction(value: Any, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    lower_ok = number >= 0 if allow_zero else number > 0
    if not lower_ok or number > 1:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ConfigError(f"'{name}' must be in the interval {interval}, got {number}.")
    return number


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError("The configuration root must be a mapping.")
    if loaded.get("version") != 1:
        raise ConfigError("Only configuration schema version 1 is supported.")

    experiment = _section(loaded, "experiment")
    dataset = _section(loaded, "dataset")
    paths = _section(loaded, "paths")
    model = _section(loaded, "model")
    selection = _section(loaded, "selection")
    generation = _section(loaded, "generation")
    training = _section(loaded, "training")
    slurm = _section(loaded, "slurm")
    smoke = loaded.setdefault("smoke", {})
    if not isinstance(smoke, dict):
        raise ConfigError("The optional 'smoke' section must be a mapping.")

    _required(experiment, "name", "experiment")
    dataset_name = str(_required(dataset, "name", "dataset")).lower()
    dataset["name"] = dataset_name
    _required(dataset, "root", "dataset")
    spec = get_dataset_spec(dataset_name)
    dataset.setdefault("num_classes", spec.num_classes)
    dataset.setdefault("image_size", spec.default_image_size)
    dataset.setdefault("ignore_index", spec.ignore_idx)
    if int(dataset["num_classes"]) != spec.num_classes:
        raise ConfigError(
            f"dataset.num_classes={dataset['num_classes']} does not match "
            f"the registered {dataset_name} value {spec.num_classes}."
        )
    if int(dataset["ignore_index"]) != spec.ignore_idx:
        raise ConfigError(
            f"dataset.ignore_index={dataset['ignore_index']} does not match "
            f"the registered {dataset_name} value {spec.ignore_idx}."
        )

    for key in ("step1_logs", "cache", "synthetic", "step4_logs"):
        _required(paths, key, "paths")

    _required(model, "encoder_name", "model")
    model.setdefault("checkpoint", "auto")

    selector = str(_required(selection, "selector", "selection"))
    if selector not in SUPPORTED_SELECTORS:
        raise ConfigError(
            f"Unsupported selection.selector '{selector}'. "
            f"Choose one of {sorted(SUPPORTED_SELECTORS)}."
        )
    selection["tau"] = _fraction(
        selection.get("tau", 0.10), "selection.tau", allow_zero=True
    )
    selection["min_obj_size"] = float(selection.get("min_obj_size", 0.0))
    selection.setdefault("selector_seed", 0)
    selection.setdefault("num_samples", -1)
    selection.setdefault("max_samples", -1)
    selection.setdefault("every_nth_sample", 1)
    selection.setdefault("transforms_per_sample", 1)
    selection.setdefault("square_cache_crops", False)
    selection.setdefault("cache_crop_size", dataset["image_size"])
    selection.setdefault("save_heatmaps", False)

    inpainter = str(_required(generation, "inpainter", "generation"))
    if inpainter not in SUPPORTED_INPAINTERS:
        raise ConfigError(
            f"Unsupported generation.inpainter '{inpainter}'. "
            f"Choose one of {sorted(SUPPORTED_INPAINTERS)}."
        )
    generation.setdefault("model_id", None)
    generation.setdefault("max_samples", selection["num_samples"])
    generation.setdefault("context_guidance_strength", 0.0)
    generation.setdefault("num_steps", 40)
    generation.setdefault("cfg_guidance_scale", 7.0)
    generation.setdefault("seed", 42)
    generation.setdefault("post_process", True)
    generation.setdefault("guidance_region", "not-selected")
    generation.setdefault("classifier_guidance_schedule", False)
    generation.setdefault("mask_erosion_kernel", 0)
    generation.setdefault("image_size", 1024)
    generation.setdefault("control_methods", [])
    generation.setdefault("controlnet_paths", [])
    generation.setdefault("controlnet_conditioning_scales", [])
    control_lengths = {
        len(generation["control_methods"]),
        len(generation["controlnet_paths"]),
        len(generation["controlnet_conditioning_scales"]),
    }
    if len(control_lengths) != 1:
        raise ConfigError(
            "generation.control_methods, controlnet_paths, and "
            "controlnet_conditioning_scales must have equal lengths."
        )
    if inpainter == "sdxl_controlnet_inpainting" and not generation[
        "control_methods"
    ]:
        raise ConfigError(
            "sdxl_controlnet_inpainting requires at least one configured control."
        )
    if inpainter in {
        "sdxl_diffusers_inpainting",
        "sd15_diffusers_inpainting",
        "flux_fill",
    } and float(generation["context_guidance_strength"]) != 0:
        raise ConfigError(
            f"{inpainter} requires generation.context_guidance_strength: 0."
        )
    if inpainter in {
        "sdxl_diffusers_inpainting",
        "sd15_diffusers_inpainting",
    } and int(generation["num_steps"]) < 2:
        raise ConfigError(
            f"{inpainter} requires generation.num_steps >= 2 with the "
            "pipeline's 0.9999 inpaint strength."
        )

    training["real_split"] = _fraction(
        training.get("real_split", 1.0), "training.real_split"
    )
    training["synthetic_split"] = _fraction(
        training.get("synthetic_split", 1.0), "training.synthetic_split"
    )
    selection["subset_split"] = _fraction(
        selection.get("subset_split", training["real_split"]),
        "selection.subset_split",
    )
    training.setdefault("baseline_seed", 42)
    training.setdefault("fine_tune_seeds", [42])
    training.setdefault("baseline_steps", 20_000)
    training.setdefault("fine_tune_steps", 20_000)
    training.setdefault("baseline_patience", 3)
    training.setdefault("fine_tune_patience", 5)
    training.setdefault("fine_tune_learning_rate", 2e-5)
    training.setdefault("batch_size", 4)
    training.setdefault("num_workers", 16)
    seeds = [int(seed) for seed in training["fine_tune_seeds"]]
    if not seeds:
        raise ConfigError("training.fine_tune_seeds must be a non-empty list.")
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ConfigError(
            "training.fine_tune_seeds must be consecutive for the Slurm wrapper."
        )
    training["fine_tune_seeds"] = seeds

    slurm.setdefault("use_docker", False)
    slurm.setdefault("conda_env", "preserve-the-hard")
    smoke.setdefault("max_samples", 2)
    smoke.setdefault("num_samples", 2)
    smoke.setdefault("train_steps", 2)
    smoke.setdefault("num_steps", 2)
    smoke.setdefault("fine_tune_runs", 1)
    smoke.setdefault("batch_size", 1)
    smoke.setdefault("num_workers", 0)

    loaded["_config_path"] = str(config_path)
    return loaded


def config_to_env(config: dict[str, Any]) -> dict[str, str]:
    dataset = config["dataset"]
    paths = config["paths"]
    model = config["model"]
    selection = config["selection"]
    generation = config["generation"]
    training = config["training"]
    slurm = config["slurm"]

    def boolean(value: Any) -> str:
        return "1" if bool(value) else "0"

    checkpoint = str(model.get("checkpoint") or "auto")
    return {
        "CONFIG_NAME": str(config["experiment"]["name"]),
        "DATASET_NAME": str(dataset["name"]),
        "REAL_ROOT": str(dataset["root"]),
        "DATA_ROOT": str(dataset["root"]),
        "NUM_CLASSES": str(dataset["num_classes"]),
        "IMAGE_SIZE": str(dataset["image_size"]),
        "ENCODER_NAME": str(model["encoder_name"]),
        "CKPT_PATH": checkpoint,
        "GUIDANCE_CHECKPOINT": checkpoint,
        "STEP1_LOGGER_SAVE_DIR": str(paths["step1_logs"]),
        "STEP4_LOGGER_SAVE_DIR": str(paths["step4_logs"]),
        "CACHE_DIR": str(paths["cache"]),
        "ROOT_DIR": str(paths["cache"]),
        "OUTPUT_FOLDER": str(paths["synthetic"]),
        "SYN_ROOT": str(paths["synthetic"]),
        "STEP1_SEED": str(training["baseline_seed"]),
        "STEP4_SEED": str(training["fine_tune_seeds"][0]),
        "NUM_RUNS": str(len(training["fine_tune_seeds"])),
        "REAL_SPLIT": str(training["real_split"]),
        "SYN_SPLIT": str(training["synthetic_split"]),
        "STEP1_TRAIN_STEPS": str(training["baseline_steps"]),
        "STEP4_TRAIN_STEPS": str(training["fine_tune_steps"]),
        "STEP1_EARLY_STOPPING_PATIENCE": str(training["baseline_patience"]),
        "STEP4_EARLY_STOPPING_PATIENCE": str(training["fine_tune_patience"]),
        "LEARNING_RATE": str(training["fine_tune_learning_rate"]),
        "BATCH_SIZE": str(training["batch_size"]),
        "NUM_WORKERS": str(training["num_workers"]),
        "SELECTOR_TYPE": str(selection["selector"]),
        "SELECTOR_SEED": str(selection["selector_seed"]),
        "MIN_PIXEL": str(selection["tau"]),
        "MIN_OBJ_SIZE": str(selection["min_obj_size"]),
        "SUBSET_SPLIT": str(selection["subset_split"]),
        "NUM_SAMPLES": str(selection["num_samples"]),
        "STEP2_MAX_SAMPLES": str(selection["max_samples"]),
        "EVERY_NTH_SAMPLE": str(selection["every_nth_sample"]),
        "TRANSFORMS_PER_SAMPLE": str(selection["transforms_per_sample"]),
        "SQUARE_CACHE_CROPS": boolean(selection["square_cache_crops"]),
        "CACHE_CROP_SIZE": str(selection["cache_crop_size"]),
        "SAVE_HEATMAPS": boolean(selection["save_heatmaps"]),
        "INPAINTER_MODEL": str(generation["inpainter"]),
        "INPAINTER_MODEL_ID": str(generation.get("model_id") or ""),
        "STEP3_MAX_SAMPLES": str(generation["max_samples"]),
        "CONTEXT_GUIDANCE_STRENGTH": str(
            generation["context_guidance_strength"]
        ),
        "NUM_STEPS": str(generation["num_steps"]),
        "CFG_GUIDANCE_SCALE": str(generation["cfg_guidance_scale"]),
        "STEP3_SEED": str(generation["seed"]),
        "POST_PROCESS": str(bool(generation["post_process"])),
        "GUIDANCE_REGION": str(generation["guidance_region"]),
        "CLASSIFIER_GUIDANCE_SCHEDULE": str(
            bool(generation["classifier_guidance_schedule"])
        ),
        "MASK_EROSION_KERNEL": str(generation["mask_erosion_kernel"]),
        "GENERATION_IMAGE_SIZE": str(generation["image_size"]),
        "CONTROL_METHODS": " ".join(
            str(value) for value in generation["control_methods"]
        ),
        "CONTROLNET_PATHS": " ".join(
            str(value) for value in generation["controlnet_paths"]
        ),
        "CONTROLNET_CONDITIONING_SCALES": " ".join(
            str(value)
            for value in generation["controlnet_conditioning_scales"]
        ),
        "USE_DOCKER": boolean(slurm["use_docker"]),
        "CONDA_ENV_NAME": str(slurm["conda_env"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and inspect a pipeline YAML config."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--format", choices=["shell", "json", "validate"], default="validate"
    )
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    if args.format == "shell":
        for key, value in config_to_env(config).items():
            print(f"export {key}={shlex.quote(value)}")
    elif args.format == "json":
        print(json.dumps(config, indent=2, sort_keys=True))
    else:
        print(f"Valid configuration: {args.config}")


if __name__ == "__main__":
    main()
