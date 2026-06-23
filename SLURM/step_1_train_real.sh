#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

# Adjust the generic GPU request above if your cluster uses another resource syntax.

ENCODER_NAME="${ENCODER_NAME:-vit_small_patch14_dinov2}"
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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

DATASET_NAME="${DATASET_NAME:-cityscapes}"
REAL_ROOT="${REAL_ROOT:-data/${DATASET_NAME}}"
LOGGER_SAVE_DIR="${LOGGER_SAVE_DIR:-training_logs/${DATASET_NAME}/real_only}"
SEED="${STEP1_SEED:-${SEED:-42}}"
REAL_SPLIT="${REAL_SPLIT:-1.00}"
TRAIN_STEPS="${STEP1_TRAIN_STEPS:-${TRAIN_STEPS:-20000}}"
EARLY_STOPPING_PATIENCE="${STEP1_EARLY_STOPPING_PATIENCE:-${EARLY_STOPPING_PATIENCE:-3}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# Set WANDB_API_KEY in the submission environment only if online logging is desired.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-preserve-the-hard}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="${WANDB_NAME:-step1-${DATASET_NAME}-train-real-${SLURM_JOB_ID:-local}}"
export WANDB_DIR="${WANDB_DIR:-training_logs/wandb}"

run_command="python -u guided_generation/main_scripts/step_1_train_real.py \
    --dataset_name ${DATASET_NAME} \
    --real_root ${REAL_ROOT} \
    --logger_save_dir ${LOGGER_SAVE_DIR} \
    --encoder_name ${ENCODER_NAME} \
    --seed ${SEED} \
    --real_split ${REAL_SPLIT} \
    --train_steps ${TRAIN_STEPS} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS}"

srun --ntasks=1 --kill-on-bad-exit=1 ./SLURM/run_command.sh "$run_command" 16g
