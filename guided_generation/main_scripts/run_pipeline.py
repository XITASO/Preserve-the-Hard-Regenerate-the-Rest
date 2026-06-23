from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guided_generation.config import ConfigError, load_pipeline_config


def latest_checkpoint(root: str | Path) -> str | None:
    candidates = [path for path in Path(root).rglob("*.ckpt") if path.is_file()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def run_command(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def python_script(name: str) -> list[str]:
    return [sys.executable, "-u", f"guided_generation/main_scripts/{name}"]


def apply_smoke_overrides(config: dict[str, Any]) -> None:
    smoke = config["smoke"]
    config["selection"]["max_samples"] = int(smoke["max_samples"])
    config["selection"]["num_samples"] = int(smoke["num_samples"])
    config["generation"]["max_samples"] = int(smoke["max_samples"])
    config["generation"]["num_steps"] = int(smoke["num_steps"])
    config["training"]["baseline_steps"] = int(smoke["train_steps"])
    config["training"]["fine_tune_steps"] = int(smoke["train_steps"])
    config["training"]["batch_size"] = int(smoke["batch_size"])
    config["training"]["num_workers"] = int(smoke["num_workers"])
    config["training"]["fine_tune_seeds"] = config["training"][
        "fine_tune_seeds"
    ][: int(smoke["fine_tune_runs"])]


def build_commands(
    config: dict[str, Any], checkpoint: str
) -> dict[int, list[list[str]]]:
    dataset = config["dataset"]
    paths = config["paths"]
    model = config["model"]
    selection = config["selection"]
    generation = config["generation"]
    training = config["training"]

    step1 = python_script("step_1_train_real.py") + [
        "--dataset_name", str(dataset["name"]),
        "--real_root", str(dataset["root"]),
        "--logger_save_dir", str(paths["step1_logs"]),
        "--encoder_name", str(model["encoder_name"]),
        "--seed", str(training["baseline_seed"]),
        "--real_split", str(training["real_split"]),
        "--train_steps", str(training["baseline_steps"]),
        "--early_stopping_patience", str(training["baseline_patience"]),
        "--batch_size", str(training["batch_size"]),
        "--num_workers", str(training["num_workers"]),
    ]
    step2 = python_script("step_2_select_samples.py") + [
        "--dataset_name", str(dataset["name"]),
        "--data_root", str(dataset["root"]),
        "--ckpt_path", checkpoint,
        "--num_classes", str(dataset["num_classes"]),
        "--image_size", str(dataset["image_size"]),
        "--encoder_name", str(model["encoder_name"]),
        "--selector_type", str(selection["selector"]),
        "--selector_seed", str(selection["selector_seed"]),
        "--min_pixel", str(selection["tau"]),
        "--min_obj_size", str(selection["min_obj_size"]),
        "--subset_split", str(selection["subset_split"]),
        "--num_samples", str(selection["num_samples"]),
        "--max_samples", str(selection["max_samples"]),
        "--every_nth_sample", str(selection["every_nth_sample"]),
        "--transforms_per_sample", str(selection["transforms_per_sample"]),
        "--cache_dir", str(paths["cache"]),
    ]
    if selection["square_cache_crops"]:
        step2 += [
            "--square_cache_crops",
            "--cache_crop_size", str(selection["cache_crop_size"]),
        ]
    if selection["save_heatmaps"]:
        step2.append("--save_heatmaps")

    step3 = python_script("step_3_generate_synthetic.py") + [
        "--dataset_name", str(dataset["name"]),
        "--root_dir", str(paths["cache"]),
        "--max_samples", str(generation["max_samples"]),
        "--context_guidance_strength",
        str(generation["context_guidance_strength"]),
        "--inpainter_model", str(generation["inpainter"]),
        "--output_folder", str(paths["synthetic"]),
        "--guidance_checkpoint", checkpoint,
        "--guide_num_classes", str(dataset["num_classes"]),
        "--image_size", str(dataset["image_size"]),
        "--encoder_name", str(model["encoder_name"]),
        "--num_steps", str(generation["num_steps"]),
        "--cfg_guidance_scale", str(generation["cfg_guidance_scale"]),
        "--seed", str(generation["seed"]),
        "--post_process", str(bool(generation["post_process"])),
        "--guidance_region", str(generation["guidance_region"]),
        "--classifier_guidance_schedule",
        str(bool(generation["classifier_guidance_schedule"])),
        "--mask_erosion_kernel", str(generation["mask_erosion_kernel"]),
        "--generation_image_size", str(generation["image_size"]),
    ]
    if generation.get("model_id"):
        step3 += ["--inpainter_model_id", str(generation["model_id"])]
    if generation["control_methods"]:
        step3 += ["--control_methods", *map(str, generation["control_methods"])]
        step3 += ["--controlnet_paths", *map(str, generation["controlnet_paths"])]
        step3 += [
            "--controlnet_conditioning_scales",
            *map(str, generation["controlnet_conditioning_scales"]),
        ]

    step4_commands = []
    for seed in training["fine_tune_seeds"]:
        metric_file = Path(paths["step4_logs"]) / f"metrics_seed_{seed}.json"
        step4_commands.append(
            python_script("step_4_train_synthetic.py")
            + [
                "--dataset_name", str(dataset["name"]),
                "--root", str(paths["synthetic"]),
                "--real_root", str(dataset["root"]),
                "--logger_save_dir", str(paths["step4_logs"]),
                "--train_steps", str(training["fine_tune_steps"]),
                "--learning_rate", str(training["fine_tune_learning_rate"]),
                "--seed", str(seed),
                "--real_split", str(training["real_split"]),
                "--syn_split", str(training["synthetic_split"]),
                "--early_stopping_patience",
                str(training["fine_tune_patience"]),
                "--ckpt_path", checkpoint,
                "--encoder_name", str(model["encoder_name"]),
                "--metric_output_file", str(metric_file),
                "--batch_size", str(training["batch_size"]),
                "--num_workers", str(training["num_workers"]),
            ]
        )
    return {1: [step1], 2: [step2], 3: [step3], 4: step4_commands}


def parse_stages(value: str) -> list[int]:
    try:
        stages = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Stages must be comma-separated integers."
        ) from exc
    if not stages or any(stage not in {1, 2, 3, 4} for stage in stages):
        raise argparse.ArgumentTypeError("Stages must be selected from 1,2,3,4.")
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the four-stage pipeline locally from YAML."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", type=parse_stages, default=[1, 2, 3, 4])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = deepcopy(load_pipeline_config(args.config))
    if args.smoke:
        apply_smoke_overrides(config)

    requested = args.checkpoint or config["model"].get("checkpoint", "auto")
    checkpoint = None if requested in {None, "", "auto"} else str(requested)
    if 1 not in args.stages and checkpoint is None:
        checkpoint = latest_checkpoint(config["paths"]["step1_logs"])
        if checkpoint is None and not args.dry_run:
            raise ConfigError(
                "No checkpoint was provided and none was found under "
                "paths.step1_logs."
            )

    commands = build_commands(config, checkpoint or "${STEP1_CHECKPOINT}")
    for stage in args.stages:
        if stage == 1:
            run_command(commands[1][0], args.dry_run)
            if not args.dry_run:
                checkpoint = latest_checkpoint(config["paths"]["step1_logs"])
                if checkpoint is None:
                    raise FileNotFoundError(
                        "Step 1 completed without producing a checkpoint under "
                        f"{config['paths']['step1_logs']}."
                    )
                commands = build_commands(config, checkpoint)
            continue
        if checkpoint is None and not args.dry_run:
            raise ConfigError(f"Step {stage} requires a baseline checkpoint.")
        for command in commands[stage]:
            run_command(command, args.dry_run)


if __name__ == "__main__":
    main()
