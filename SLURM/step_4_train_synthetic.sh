#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

# Adjust the generic GPU request above if your cluster uses another resource syntax.

set -e

ENCODER_NAME="${ENCODER_NAME:-vit_small_patch14_dinov2}"

NUM_RUNS="${NUM_RUNS:-5}"
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
        --num_runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --num_runs=*)
            NUM_RUNS="${1#*=}"
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
SYN_ROOT="${SYN_ROOT:-synthetic_data/${DATASET_NAME}/iteration1}"
LOGGER_SAVE_DIR="${LOGGER_SAVE_DIR:-training_logs/${DATASET_NAME}/iteration1}"
CKPT_PATH_REQUESTED="${CKPT_PATH:-}"
CKPT_PATH="${CKPT_PATH:-auto}"
CKPT_SEARCH_DIR="${CKPT_SEARCH_DIR:-}"
SEED="${STEP4_SEED:-${SEED:-42}}"
REAL_SPLIT="${REAL_SPLIT:-1.0}"
SYN_SPLIT="${SYN_SPLIT:-1.00}"
TRAIN_STEPS="${STEP4_TRAIN_STEPS:-${TRAIN_STEPS:-20000}}"
LEARNING_RATE="${LEARNING_RATE:-}"
EARLY_STOPPING_PATIENCE="${STEP4_EARLY_STOPPING_PATIENCE:-${EARLY_STOPPING_PATIENCE:-5}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"

if [ -n "$CKPT_SEARCH_DIR" ] && { [ -z "$CKPT_PATH_REQUESTED" ] || [ "$CKPT_PATH_REQUESTED" = "auto" ]; }; then
    CKPT_PATH="$(find "$CKPT_SEARCH_DIR" -path "*/checkpoints/*.ckpt" -type f -printf "%T@ %p\n" | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [ -z "$CKPT_PATH" ]; then
        echo "ERROR: Could not find a Step 1 checkpoint under CKPT_SEARCH_DIR=$CKPT_SEARCH_DIR"
        exit 1
    fi
    echo "Resolved CKPT_PATH from CKPT_SEARCH_DIR: $CKPT_PATH"
fi

if [ "$CKPT_PATH_REQUESTED" = "auto" ] && [ -z "$CKPT_SEARCH_DIR" ]; then
    CKPT_PATH=""
    echo "CKPT_PATH=auto requested without CKPT_SEARCH_DIR; letting Python resolve from synthetic metadata."
fi

# Set WANDB_API_KEY in the submission environment only if online logging is desired.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-preserve-the-hard}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="${WANDB_NAME:-step4-${DATASET_NAME}-train-synthetic-${SLURM_JOB_ID:-local}}"
export WANDB_DIR="${WANDB_DIR:-training_logs/wandb}"

RESULTS_DIR="training_logs/step4_multirun_${SLURM_JOB_ID:-$$}"
mkdir -p "${RESULTS_DIR}"

echo "========================================"
echo "Step 4: Training with ${NUM_RUNS} run(s)"
echo "Base seed: ${SEED}  |  Dataset: ${DATASET_NAME}"
echo "Results dir: ${RESULTS_DIR}"
echo "========================================"

for run_idx in $(seq 1 ${NUM_RUNS}); do
    RUN_SEED=$((SEED + run_idx - 1))
    METRIC_FILE="${RESULTS_DIR}/run_${run_idx}.json"

    echo ""
    echo "--- RUN ${run_idx}/${NUM_RUNS}  seed=${RUN_SEED} ---"

    run_command="python -u guided_generation/main_scripts/step_4_train_synthetic.py \
        --dataset_name ${DATASET_NAME} \
        --real_root ${REAL_ROOT} \
        --root ${SYN_ROOT} \
        --encoder_name ${ENCODER_NAME} \
        --seed ${RUN_SEED} \
        --real_split ${REAL_SPLIT} \
        --syn_split ${SYN_SPLIT} \
        --logger_save_dir ${LOGGER_SAVE_DIR} \
        --train_steps ${TRAIN_STEPS} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --metric_output_file ${METRIC_FILE} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS}"

    if [ -n "$LEARNING_RATE" ]; then
        run_command="$run_command --learning_rate ${LEARNING_RATE}"
    fi

    if [ -n "$CKPT_PATH" ]; then
        run_command="$run_command --ckpt_path ${CKPT_PATH}"
    fi

    srun --ntasks=1 --kill-on-bad-exit=1 ./SLURM/run_command.sh "$run_command" 16g

    echo "--- RUN ${run_idx}/${NUM_RUNS} DONE ---"
done

# ── Aggregate statistics across all runs ──────────────────────────────────────
echo ""
echo "========================================"
echo "MULTI-RUN SUMMARY  (${NUM_RUNS} run(s))"
echo "========================================"

python3 - "${RESULTS_DIR}" "${NUM_RUNS}" <<'PYEOF'
import sys, json, os, glob, math

results_dir = sys.argv[1]
num_runs    = int(sys.argv[2])

files = sorted(glob.glob(os.path.join(results_dir, "run_*.json")))
if not files:
    print("No metric files found — runs may have failed.")
    sys.exit(1)

# Collect all metric keys present across runs
all_metrics: dict[str, list[float]] = {}
metadata: dict[str, list[str]] = {}
seeds: list[int] = []

for f in files:
    with open(f) as fh:
        data = json.load(fh)
    seeds.append(data.get("seed", -1))
    for k, v in data.items():
        if k == "seed":
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            all_metrics.setdefault(k, []).append(float(v))
        else:
            metadata.setdefault(k, []).append(str(v))

# Per-run table
header = f"{'Run':>4}  {'Seed':>6}  " + "  ".join(f"{k:>18}" for k in all_metrics)
print(header)
print("-" * len(header))
for i, (seed, f) in enumerate(zip(seeds, files)):
    with open(f) as fh:
        data = json.load(fh)
    def fmt_metric(key: str) -> str:
        value = data.get(key, float("nan"))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{float(value):>18.6f}"
        return f"{'NA':>18}"
    row = f"{i+1:>4}  {seed:>6}  " + "  ".join(
        fmt_metric(k) for k in all_metrics
    )
    print(row)

if metadata:
    print()
    print("Metadata:")
    for i, (seed, f) in enumerate(zip(seeds, files)):
        with open(f) as fh:
            data = json.load(fh)
        for key in metadata:
            if key in data:
                print(f"run={i+1} seed={seed} {key}={data[key]}")

# Summary statistics
print()
print(f"{'Metric':<25}  {'N':>4}  {'Mean':>10}  {'Std':>10}  {'Variance':>12}  {'Min':>10}  {'Max':>10}")
print("-" * 85)
for metric, values in all_metrics.items():
    n    = len(values)
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / n  if n > 1 else 0.0
    std  = math.sqrt(var)
    print(f"{metric:<25}  {n:>4}  {mean:>10.6f}  {std:>10.6f}  {var:>12.8f}  {min(values):>10.6f}  {max(values):>10.6f}")

PYEOF
