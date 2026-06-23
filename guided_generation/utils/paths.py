from pathlib import Path
from typing import Union


def find_unique_checkpoint(output_folder: Union[str, Path]) -> str:
    """
    Given an output folder (e.g. "training_logs/cityscapes/temp"), find the unique
    PyTorch Lightning checkpoint under:
        <output_folder>/lightning_logs/<run_id>/checkpoints/*.ckpt

    Ensures there is exactly one run dir under lightning_logs and exactly one .ckpt file
    in its checkpoints directory. Returns the absolute path to that checkpoint.

    Raises:
        FileNotFoundError or RuntimeError with a helpful message if requirements are not met.
    """
    out = Path(output_folder)
    lightning = out / "lightning_logs"
    if not lightning.exists() or not lightning.is_dir():
        raise FileNotFoundError(f"Missing directory: {lightning}")

    runs = [d for d in lightning.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found under {lightning}")
    if len(runs) > 1:
        raise RuntimeError(f"Multiple run directories found under {lightning}: {[str(r.name) for r in runs]}")

    run_dir = runs[0]
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoints directory: {ckpt_dir}")

    ckpts = [p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix == ".ckpt"]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    if len(ckpts) > 1:
        raise RuntimeError(f"Multiple .ckpt files found in {ckpt_dir}: {[p.name for p in ckpts]}")

    return str(ckpts[0].resolve())
