#!/bin/bash
#
# Submit steps 1-4 as a chained SLURM pipeline.
# Each step runs as its own job; step N+1 starts only after step N exits 0
# (--dependency=afterok). Checkpoint paths and synthetic-data paths are
# shared between steps via environment variables. Set CKPT_PATH explicitly or
# set CKPT_SEARCH_DIR to the Step 1 logger directory before running Step 2.
#
# Usage:
#   ./SLURM/run_pipeline.sh configs/cityscapes.yaml
#
# Optional env vars:
#   PIPELINE_TAG       — short label appended to log dirs (default: timestamp)
#   SKIP_STEP1=1       — skip step 1 (requires CKPT_PATH or STEP1_LOGGER_SAVE_DIR pointing at a prior run)
#   SKIP_STEP2=1, SKIP_STEP3=1, SKIP_STEP4=1
#   Any per-step var (e.g. TRAIN_STEPS, MAX_SAMPLES) is forwarded to the
#   underlying step scripts via sbatch --export=ALL.

set -euo pipefail

CONFIG_PATH="${1:-${CONFIG_PATH:-configs/cityscapes.yaml}}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

eval "$(python -m guided_generation.config --config "$CONFIG_PATH" --format shell)"
export CONFIG_PATH
export CKPT_SEARCH_DIR="${CKPT_SEARCH_DIR:-$STEP1_LOGGER_SAVE_DIR}"

echo "=== Pipeline configuration ==="
echo "  CONFIG:                 $CONFIG_PATH"
echo "  DATASET_NAME:           $DATASET_NAME"
echo "  ENCODER_NAME:           $ENCODER_NAME"
echo "  PIPELINE_TAG:           $PIPELINE_TAG"
echo "  STEP1_LOGGER_SAVE_DIR:  $STEP1_LOGGER_SAVE_DIR"
echo "  CACHE_DIR (step2 out):  $CACHE_DIR"
echo "  CACHE_CROP_SIZE:        ${CACHE_CROP_SIZE:-dataset default}"
echo "  OUTPUT_FOLDER (step3):  $OUTPUT_FOLDER"
echo "  LOGGER_SAVE_DIR (step4):$STEP4_LOGGER_SAVE_DIR"
echo "==============================="

# Helper: submit a step with optional dependency, return job id on stdout.
submit_step() {
    local job_name="$1"; shift
    local script="$1"; shift
    local dep_jobid="$1"; shift
    local dep_arg=()
    if [ -n "$dep_jobid" ]; then
        dep_arg=(--dependency=afterok:"$dep_jobid")
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "sbatch --job-name=$job_name ${dep_arg[*]} --export=ALL $script $*" >&2
        echo "dry-run-${job_name}"
        return
    fi
    # --export=ALL forwards the current shell's env vars (DATASET_NAME, ENCODER_NAME,
    # STEP1_LOGGER_SAVE_DIR, CACHE_DIR, OUTPUT_FOLDER, SYN_ROOT, ...) into the job.
    local out
    out=$(sbatch --parsable --job-name="$job_name" "${dep_arg[@]}" --export=ALL "$script" "$@")
    echo "$out"
}

JOB1=""
JOB2=""
JOB3=""
JOB4=""

if [ "${SKIP_STEP1:-0}" != "1" ]; then
    export LOGGER_SAVE_DIR="$STEP1_LOGGER_SAVE_DIR"
    JOB1=$(submit_step "step1-${DATASET_NAME}-${PIPELINE_TAG}" SLURM/step_1_train_real.sh "" --encoder_name "$ENCODER_NAME")
    echo "Submitted step 1: job $JOB1"
fi

# Steps 2 and 3 don't use LOGGER_SAVE_DIR; unset to avoid leaking step 1's value.
unset LOGGER_SAVE_DIR

if [ "${SKIP_STEP2:-0}" != "1" ]; then
    JOB2=$(submit_step "step2-${DATASET_NAME}-${PIPELINE_TAG}" SLURM/step_2_select_samples.sh "$JOB1" --encoder_name "$ENCODER_NAME")
    echo "Submitted step 2: job $JOB2 (after $JOB1)"
fi

if [ "${SKIP_STEP3:-0}" != "1" ]; then
    JOB3=$(submit_step "step3-${DATASET_NAME}-${PIPELINE_TAG}" SLURM/step_3_generate_synthetic.sh "$JOB2" --encoder_name "$ENCODER_NAME")
    echo "Submitted step 3: job $JOB3 (after $JOB2)"
fi

if [ "${SKIP_STEP4:-0}" != "1" ]; then
    export LOGGER_SAVE_DIR="$STEP4_LOGGER_SAVE_DIR"
    JOB4=$(submit_step "step4-${DATASET_NAME}-${PIPELINE_TAG}" SLURM/step_4_train_synthetic.sh "$JOB3" --encoder_name "$ENCODER_NAME")
    echo "Submitted step 4: job $JOB4 (after $JOB3)"
fi

echo ""
echo "Pipeline submitted. Track with: squeue -u \$USER -n step1-${DATASET_NAME}-${PIPELINE_TAG},step2-${DATASET_NAME}-${PIPELINE_TAG},step3-${DATASET_NAME}-${PIPELINE_TAG},step4-${DATASET_NAME}-${PIPELINE_TAG}"
echo "Logs use the Slurm pattern: slurm-<job-name>-<jobid>.out"
