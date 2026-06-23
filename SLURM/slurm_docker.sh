#!/bin/bash

# Generic optional Docker wrapper for Slurm jobs. Configure the image and any
# dataset mount in the submission environment; no cluster filesystem layout is
# assumed here.
set -euo pipefail

command_to_run="${1:?Usage: slurm_docker.sh '<command>' [shared-memory-size]}"
shared_memory_size="${2:-16g}"
: "${SLURM_DOCKER_IMAGE:?Set SLURM_DOCKER_IMAGE to a locally available image.}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_workdir="${CONTAINER_WORKDIR:-/workspace/preserve_the_hard}"

gpu_args=(--gpus all)
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    gpu_args=(--gpus "device=${SLURM_JOB_GPUS}")
fi

dataset_mount_args=()
if [ -n "${DATASET_HOST_PATH:-}" ]; then
    : "${DATASET_CONTAINER_PATH:?Set DATASET_CONTAINER_PATH with DATASET_HOST_PATH.}"
    dataset_mount_args=(
        --mount "type=bind,source=${DATASET_HOST_PATH},destination=${DATASET_CONTAINER_PATH},readonly"
    )
fi

docker run --rm \
    "${gpu_args[@]}" \
    --mount "type=bind,source=${repo_root},destination=${container_workdir}" \
    "${dataset_mount_args[@]}" \
    --user "$(id -u):$(id -g)" \
    --network host \
    --shm-size "${shared_memory_size}" \
    --env PYTHONUNBUFFERED=1 \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env WANDB_API_KEY \
    --env WANDB_PROJECT \
    --env WANDB_ENTITY \
    --env WANDB_NAME \
    --env WANDB_MODE \
    --env WANDB_DIR \
    --env HF_TOKEN \
    --env HUGGING_FACE_HUB_TOKEN \
    --env HF_HUB_TOKEN \
    -w "${container_workdir}" \
    "${SLURM_DOCKER_IMAGE}" \
    bash -lc "${command_to_run}"
