#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

# Adjust the generic GPU request above if your cluster uses another resource syntax.

ENCODER_NAME="${ENCODER_NAME:-vit_small_patch14_dinov2}"

SIMPLE_MODE="${SIMPLE_MODE:-0}"
SIMPLE_MODE_EROSION_KERNEL="${SIMPLE_MODE_EROSION_KERNEL:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder_name)
            ENCODER_NAME="$2"
            shift 2
            ;;
        --encoder_name=*)
            ENCODER_NAME="${1#*=}"
            shift
            ;;
        --simple_mode)
            SIMPLE_MODE=1
            shift
            ;;
        --simple_mode_erosion_kernel)
            SIMPLE_MODE_EROSION_KERNEL="$2"
            shift 2
            ;;
        --simple_mode_erosion_kernel=*)
            SIMPLE_MODE_EROSION_KERNEL="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

DATASET_NAME="${DATASET_NAME:-cityscapes}"
DATA_ROOT="${DATA_ROOT:-data/${DATASET_NAME}}"
CKPT_PATH_REQUESTED="${CKPT_PATH:-}"
CKPT_PATH="${CKPT_PATH:-auto}"
CKPT_SEARCH_DIR="${CKPT_SEARCH_DIR:-}"
CACHE_DIR="${CACHE_DIR:-.cached_images4gen/${DATASET_NAME}/10%_masking}"
SQUARE_CACHE_CROPS="${SQUARE_CACHE_CROPS:-0}"
SELECTOR_TYPE="${SELECTOR_TYPE:-highest_entropy_class_multi}"
SELECTOR_SEED="${SELECTOR_SEED:-0}"
MIN_PIXEL="${MIN_PIXEL:-0.10}"
MIN_OBJ_SIZE="${MIN_OBJ_SIZE:-0.0}"
TRANSFORMS_PER_SAMPLE="${TRANSFORMS_PER_SAMPLE:-1}"
EVERY_NTH_SAMPLE="${EVERY_NTH_SAMPLE:-1}"
SUBSET_SPLIT="${SUBSET_SPLIT:-${REAL_SPLIT:-1.0}}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
MAX_SAMPLES="${STEP2_MAX_SAMPLES:-${MAX_SAMPLES:--1}}"
SAVE_HEATMAPS="${SAVE_HEATMAPS:-0}"

if [ -n "$CKPT_SEARCH_DIR" ] && { [ -z "$CKPT_PATH_REQUESTED" ] || [ "$CKPT_PATH_REQUESTED" = "auto" ]; }; then
    CKPT_PATH="$(find "$CKPT_SEARCH_DIR" -path "*/checkpoints/*.ckpt" -type f -printf "%T@ %p\n" | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [ -z "$CKPT_PATH" ]; then
        echo "ERROR: Could not find a Step 1 checkpoint under CKPT_SEARCH_DIR=$CKPT_SEARCH_DIR"
        exit 1
    fi
    echo "Resolved CKPT_PATH from CKPT_SEARCH_DIR: $CKPT_PATH"
fi

ckpt_path_arg=""
if [ -n "$CKPT_PATH" ]; then
    ckpt_path_arg="--ckpt_path ${CKPT_PATH}"
fi

run_command="python guided_generation/main_scripts/step_2_select_samples.py \
    --dataset_name ${DATASET_NAME} \
    --data_root ${DATA_ROOT} \
    ${ckpt_path_arg} \
    --encoder_name ${ENCODER_NAME} \
    --selector_type ${SELECTOR_TYPE} \
    --selector_seed ${SELECTOR_SEED} \
    --min_pixel ${MIN_PIXEL} \
    --min_obj_size ${MIN_OBJ_SIZE} \
    --transforms_per_sample ${TRANSFORMS_PER_SAMPLE} \
    --every_nth_sample ${EVERY_NTH_SAMPLE} \
    --subset_split ${SUBSET_SPLIT} \
    --num_samples ${NUM_SAMPLES} \
    --max_samples ${MAX_SAMPLES} \
    --cache_dir ${CACHE_DIR}"

if [ "$SAVE_HEATMAPS" = "1" ] || [ "$SAVE_HEATMAPS" = "true" ] || [ "$SAVE_HEATMAPS" = "True" ]; then
    run_command="$run_command --save_heatmaps"
fi

if [ "$SQUARE_CACHE_CROPS" = "1" ] || [ "$SQUARE_CACHE_CROPS" = "true" ] || [ "$SQUARE_CACHE_CROPS" = "True" ]; then
    run_command="$run_command --square_cache_crops --cache_crop_size ${CACHE_CROP_SIZE:-1024}"
fi

if [ "$SIMPLE_MODE" = "1" ] || [ "$SIMPLE_MODE" = "true" ] || [ "$SIMPLE_MODE" = "True" ]; then
    run_command="$run_command --simple_mode --simple_mode_erosion_kernel ${SIMPLE_MODE_EROSION_KERNEL}"
fi

srun --ntasks=1 --kill-on-bad-exit=1 ./SLURM/run_command.sh "$run_command" 16g
