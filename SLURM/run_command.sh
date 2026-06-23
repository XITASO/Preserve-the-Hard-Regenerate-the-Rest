#!/bin/bash
set -euo pipefail

command_to_run="${1:?Usage: run_command.sh '<command>' [shared-memory-size]}"
shared_memory_size="${2:-16g}"

if [ "${USE_DOCKER:-0}" = "1" ]; then
    exec ./SLURM/slurm_docker.sh "$command_to_run" "$shared_memory_size"
fi

if [ -n "${CONDA_ENV_NAME:-}" ]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: CONDA_ENV_NAME is set but conda is unavailable." >&2
        exit 1
    fi
    exec conda run --no-capture-output -n "$CONDA_ENV_NAME" \
        bash -lc "$command_to_run"
fi

exec bash -lc "$command_to_run"
