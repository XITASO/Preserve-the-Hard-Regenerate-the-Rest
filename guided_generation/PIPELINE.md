# Pipeline map (Phase 0)

This document records the behavior of the live research pipeline as inspected in
the read-only reference repository. It describes what the current scripts
actually do; it is not yet a cleaned public interface.

## Scope note

The confirmed release root is `preserve_the_hard/`. The
`guided_generation/` package depends on the root-level `dataset_configs.py`,
`vfm4ss/`, and `SLURM/` directories, so those components are part of the same
release artifact.

## End-to-end data flow

```text
real images + real labels
        |
        v
Step 1: train baseline segmenter
        |
        +-- training logs + best .ckpt
        |
        v
Step 2: score predictive entropy and select preserve regions
        |
        +-- cache/images/train
        +-- cache/annotations/train
        +-- cache/masks/train              (white/1 = preserve)
        +-- cache/pipeline_metadata.json
        |
        v
Step 3: invert preserve mask, inpaint, paste preserved pixels back
        |
        +-- synthetic/images/train
        +-- synthetic/annotations/train    (generated pixels = ignore index)
        +-- synthetic/pipeline_metadata.json
        |
        v
Step 4: fine-tune on real + synthetic samples
        |
        +-- training logs + best .ckpt + validation metrics
```

For another iteration, use the improved Step 4 checkpoint as the Step 2
checkpoint, then rerun Steps 2--4 into new cache/output directories. No current
script performs this iterative loop automatically.

## Step 1 -- train the baseline segmenter

### Entry points

- Slurm: `SLURM/step_1_train_real.sh`
- Python: `guided_generation/main_scripts/step_1_train_real.py`
- Trainer: `vfm4ss/segmentation_trainer.py`

The Slurm wrapper builds a Python command and runs it through
`SLURM/slurm_docker.sh`. The Python entry point creates a
`SemanticSegmentationTrainer` for the real-data dataset variant and calls
`train()`.

### Model and optimization

- Default encoder: `vit_small_patch14_dinov2` through `timm`.
- Decoder: `vfm4ss.models.linear_decoder.LinearDecoder`.
- Loss: per-pixel cross entropy with the dataset ignore index.
- Default learning rate from scratch: `1e-4`; encoder multiplier: `0.1`.
- Lightning defaults in the trainer: batch size 4, four-step gradient
  accumulation, mixed precision, validation every 1,000 optimizer steps, and a
  best-checkpoint monitor on `val_0_miou`.
- `--encoder_name` permits another compatible `timm` backbone (for example a
  ResNet or ImageNet ViT), but the checkpoint and later guide must use the same
  architecture.

The public release retains the canonical DINO/timm encoder plus linear decoder
path. Separate baseline implementations were excluded during Phase 1.

### Inputs

- `--dataset_name`: `cityscapes`, `uavid`, `pascal_voc`, `cocostuff10k`,
  `bdd100k`, or `ade20k`.
- `--real_root`: root of the real dataset.
- `--real_split`: deterministic fraction of real training pairs selected using
  `FIXED_SUBSET_SEED = 44`.
- `--encoder_name`, `--seed`, `--train_steps`, and
  `--early_stopping_patience`.

Cityscapes uses its native `leftImg8bit/{train,val}` and `gtFine/{train,val}`
layout. Other datasets are loaded through the folder-based dataset spec in
`dataset_configs.py`; their image/annotation directories and suffixes come from
that spec.

### Outputs

- Lightning/W&B and CSV logs below `--logger_save_dir`.
- Best checkpoint below the logger's run-specific `checkpoints/` directory.
- Validation metrics, dataset statistics, and the selected training file list.

Step 1 does not write a pipeline metadata file. Its checkpoint path therefore
has to be passed to Step 2 explicitly or found by a wrapper.

## Step 2 -- entropy scoring and preserve-mask construction

### Entry points

- Slurm: `SLURM/step_2_select_samples.sh`
- Python: `guided_generation/main_scripts/step_2_select_samples.py`
- Guide: `guided_generation/guidance/dino_guide.py`
- Selection logic: `guided_generation/sample_selection/sample_selector.py`

The real training subset is loaded with the same fixed subset seed as Step 1.
The Step 1 checkpoint is loaded into the same encoder/linear-decoder model and
used to predict per-pixel logits.

### Uncertainty and the paper's tau parameter

For the default `highest_entropy_class_multi` selector:

1. Compute Shannon entropy at every pixel from the model logits.
2. For every ground-truth class present in the image, excluding the ignore
   index and classes smaller than `--min_obj_size`, compute mean pixel entropy.
3. Sort classes from highest to lowest mean entropy.
4. Add whole classes to the preserve set until their union covers more than
   `--min_pixel` of the image.
5. Rank images by the area-weighted mean entropy of their selected classes and,
   when `--num_samples` is positive, retain the hardest samples.

`--min_pixel` is therefore the implementation of the paper's area budget
`tau`. Because selection adds complete semantic classes and stops only after
crossing the threshold, the final preserve area can exceed tau.

Other implemented selectors are `highest_entropy_class`,
`lowest_entropy_class_multi`, `random_class_multi`, and
`random_square_region`. `--simple_mode` is a separate class-based ablation that
does not load the baseline model or compute entropy.

### Inputs and important flags

- Real dataset: `--dataset_name`, `--data_root`, `--subset_split`,
  `--every_nth_sample`, and `--max_samples`.
- Baseline: `--ckpt_path`, `--encoder_name`, optional dataset-specific
  `--num_classes` and `--image_size`.
- Selection: `--selector_type`, `--selector_seed`, `--min_pixel` (tau),
  `--min_obj_size`, `--transforms_per_sample`, and `--num_samples`.
- Cache geometry: `--square_cache_crops` and `--cache_crop_size`; used for
  high-resolution data such as UAVID.
- Diagnostics: `--save_heatmaps`.

### Outputs

`--cache_dir` receives:

```text
cache_dir/
├── images/train/       # selected real images or replayed crops
├── annotations/train/  # corresponding ground-truth labels
├── masks/train/        # preserve masks: white/1 = keep, black/0 = regenerate
├── heatmaps/train/     # optional rendered entropy maps
└── pipeline_metadata.json
```

The metadata records the dataset, source root, Step 1 checkpoint, encoder,
guide resolution, selector, tau-related settings, crop mode, and mask suffix.

## Step 3 -- inpaint the complement and construct synthetic labels

### Entry points

- Slurm: `SLURM/step_3_generate_synthetic.sh`
- Python: `guided_generation/main_scripts/step_3_generate_synthetic.py`
- Inpainter: `guided_generation/diffusion/guided_image_inpainter.py`
- Backend construction: `guided_generation/diffusion/pipeline_builder.py`

The cached mask is a preserve mask. Step 3 inverts it before calling the
inpainting backend, so the complement is regenerated.

### Default method path

- Default backend: custom SDXL inpainting using
  `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`.
- The current Slurm default sets `CONTEXT_GUIDANCE_STRENGTH=0.0`, so the
  segmentation guide is not used during denoising by default. Positive values
  enable the custom entropy-gradient callback and require a valid checkpoint.
- ControlNets are disabled in the current Step 3 Slurm wrapper because the
  control-method lists are empty.
- Inputs are resized to the nearest SDXL aspect-ratio bucket, generated, then
  resized back to native resolution.
- The exact inpainting mask consumed by the diffusion pipeline is tracked back
  to native resolution.
- With `--post_process True` (the default), original pixels are pasted back over
  preserved regions with a narrow feathered boundary.
- Generated pixels in the output annotation are replaced by the dataset ignore
  index. Thus Step 4's loss is evaluated only on labels retained from the real
  image.

### Backend and guidance choices

`--inpainter_model` currently accepts `sdxl_inpainting`,
`sdxl_controlnet_inpainting`, `sdxl_diffusers_inpainting`,
`sd15_diffusers_inpainting`, and `flux_fill`. `--inpainter_model_id` overrides
the backend's default Hugging Face model ID/path. Native Diffusers and FLUX
backends require `--context_guidance_strength 0.0`; the custom segmentation
guidance callback is implemented only for the custom SDXL path.

Other important flags are `--num_steps`, `--cfg_guidance_scale`, `--seed`,
`--generation_image_size`, `--mask_erosion_kernel`, `--base_prompt`,
`--negative_prompt`, `--guidance_region`, and
`--classifier_guidance_schedule`. Multi-ControlNet runs use aligned
`--control_methods`, `--controlnet_paths`, and
`--controlnet_conditioning_scales` lists.

### Inputs

- `--root_dir`: Step 2 cache root.
- `--dataset_name` and `--max_samples`.
- Optional `--guidance_checkpoint`; if absent, resolution first checks Step 2
  metadata and then searches `training_logs/`.
- Generation/backend flags listed above.

### Outputs

```text
output_folder/
├── images/train/              # synthetic RGB images
├── annotations/train/         # retained labels; generated region = ignore
└── pipeline_metadata.json     # source cache, checkpoint, encoder and backend
```

## Step 4 -- fine-tune on real plus synthetic data

### Entry points

- Slurm: `SLURM/step_4_train_synthetic.sh`
- Python: `guided_generation/main_scripts/step_4_train_synthetic.py`
- Trainer and loss: `vfm4ss/segmentation_trainer.py` and
  `vfm4ss/training/linear_semantic.py`

The data module builds the deterministic real subset and synthetic subset,
concatenates them, and shuffles the combined dataset. `--real_split` and
`--syn_split` are fractions of the available datasets, not an independent
batch-sampling ratio. The effective synthetic:real ratio is therefore the two
resulting sample counts.

The model is initialized from `--ckpt_path` when supplied. If it is omitted,
the Python resolver checks the synthetic metadata for the Step 1/guidance
checkpoint and then searches `training_logs/`. Fine-tuning defaults to learning
rate `2e-5`; training from scratch defaults to `1e-4`. The encoder must match
the checkpoint. Cross-entropy uses the dataset ignore index, so labels masked
by Step 3 do not contribute to the loss.

### Inputs and important flags

- `--dataset_name`, `--real_root`, and required `--root` synthetic root.
- `--real_split` and `--syn_split`.
- `--ckpt_path`, `--encoder_name`, `--learning_rate`, `--train_steps`,
  `--seed`, and `--early_stopping_patience`.
- `--metric_output_file` for a machine-readable result.

The Slurm wrapper runs `NUM_RUNS` seeds sequentially (default five), writes one
JSON result per run, then prints aggregate mean/std/min/max statistics.

### Outputs

- Fine-tuning logs and best checkpoints below `--logger_save_dir`.
- Real-validation metrics (including initial, final, and best mIoU).
- Per-run JSON files below `training_logs/step4_multirun_<job-id>/` when using
  the Slurm wrapper.

## Orchestration paths

### Slurm chain

`SLURM/run_pipeline.sh` submits the four wrappers with `afterok` dependencies
and loads dataset paths, backbone, selector, inpainter, seeds, and output paths
from one validated YAML file. Step 2 searches the configured Step 1 log root
after its dependency succeeds. Step 3 reads the checkpoint from Step 2 metadata,
and Step 4 either searches the same Step 1 root or reads synthetic metadata.

The default execution path is `conda run` through `SLURM/run_command.sh`.
Docker remains optional. Use `DRY_RUN=1` to preview all four `sbatch`
submissions without submitting jobs.

### Direct/local chain

`guided_generation/main_scripts/run_pipeline.py` translates the same YAML into
the existing Step 1--4 CLI calls. It supports stage selection, checkpoint
resume, a small smoke profile, and command-only dry runs. The legacy
`full_process.py` name now delegates to this runner.

Example configurations matching the paper are provided for Cityscapes,
BDD100K, and UAVID under `configs/`. See `CONFIGURATION.md` and
`DATASET_CONTRACT.md`.

## Required code retained

The release root contains the complete core chain: `guided_generation/`,
`dataset_configs.py`, `vfm4ss/`, the four stage launchers, orchestration script,
dependency metadata, and focused tests. Phase 1 removed generated artifacts,
excluded baselines, figure/evaluation scratch scripts, and legacy utilities;
the exact inventory is in `CLEANUP_REPORT.md`.

## Phase 1 status and remaining human decisions

- Generated checkpoints, logs, caches, bytecode, and sweep bookkeeping were
  removed and covered by `.gitignore`.
- User- and cluster-specific paths were replaced with relative defaults or
  environment variables.
- The embedded W&B credential was removed. It still needs to be revoked or
  rotated by its owner because repository cleanup cannot invalidate a secret.
- Mask2Former, SegFormer, FreeMask, and instance-augmentation baselines were
  explicitly excluded from this release.
- Confirm whether the eventual paper artifact must remain anonymous before
  adding authors, affiliations, acknowledgments, or citation metadata.
