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
ROOT_DIR="${ROOT_DIR:-.cached_images4gen/${DATASET_NAME}/iteration1}"
MAX_SAMPLES="${STEP3_MAX_SAMPLES:-${MAX_SAMPLES:-500}}"
CONTEXT_GUIDANCE_STRENGTH="${CONTEXT_GUIDANCE_STRENGTH:-0.0}"
INPAINTER_MODEL="${INPAINTER_MODEL:-sdxl_inpainting}"
INPAINTER_MODEL_ID="${INPAINTER_MODEL_ID:-}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-synthetic_data/${DATASET_NAME}/iteration1}"
GUIDANCE_CHECKPOINT_REQUESTED="${GUIDANCE_CHECKPOINT:-}"
GUIDANCE_CHECKPOINT="${GUIDANCE_CHECKPOINT:-auto}"
if [ "$GUIDANCE_CHECKPOINT_REQUESTED" = "auto" ]; then
    GUIDANCE_CHECKPOINT=""
fi
NUM_STEPS="${NUM_STEPS:-40}"
CFG_GUIDANCE_SCALE="${CFG_GUIDANCE_SCALE:-7.0}"
SEED="${SEED:-42}"
POST_PROCESS="${POST_PROCESS:-True}"
LABEL_NOISE_CONFIG="${LABEL_NOISE_CONFIG:-}"
GUIDANCE_REGION="${GUIDANCE_REGION:-not-selected}"
CLASSIFIER_GUIDANCE_SCHEDULE="${CLASSIFIER_GUIDANCE_SCHEDULE:-False}"
SEG_CONTROLNET_PATH="${SEG_CONTROLNET_PATH:-none}"
MASK_EROSION_KERNEL="${MASK_EROSION_KERNEL:-0}"
GENERATION_IMAGE_SIZE="${GENERATION_IMAGE_SIZE:-1024}"

control_methods=()
controlnet_paths=()
controlnet_conditioning_scales=()
if [ -n "${CONTROL_METHODS:-}" ]; then
    read -r -a control_methods <<< "$CONTROL_METHODS"
    read -r -a controlnet_paths <<< "${CONTROLNET_PATHS:-}"
    read -r -a controlnet_conditioning_scales <<< "${CONTROLNET_CONDITIONING_SCALES:-}"
fi

controlnet_args=""
if [ ${#control_methods[@]} -gt 0 ]; then
    controlnet_args+=" --control_methods ${control_methods[*]}"
    controlnet_args+=" --controlnet_paths ${controlnet_paths[*]}"
    controlnet_args+=" --controlnet_conditioning_scales ${controlnet_conditioning_scales[*]}"
fi

guidance_checkpoint_arg=""
if [ -n "$GUIDANCE_CHECKPOINT" ]; then
    guidance_checkpoint_arg="--guidance_checkpoint ${GUIDANCE_CHECKPOINT}"
fi

label_noise_config_arg=""
if [ -n "$LABEL_NOISE_CONFIG" ]; then
    label_noise_config_arg="--label_noise_config ${LABEL_NOISE_CONFIG}"
fi

inpainter_model_id_arg=""
if [ -n "$INPAINTER_MODEL_ID" ]; then
    inpainter_model_id_arg="--inpainter_model_id ${INPAINTER_MODEL_ID}"
fi

run_command="python guided_generation/main_scripts/step_3_generate_synthetic.py \
    --dataset_name ${DATASET_NAME} \
    --root_dir ${ROOT_DIR} \
    --max_samples ${MAX_SAMPLES} \
    --context_guidance_strength ${CONTEXT_GUIDANCE_STRENGTH} \
    --inpainter_model ${INPAINTER_MODEL} \
    ${inpainter_model_id_arg} \
    --output_folder ${OUTPUT_FOLDER} \
    ${guidance_checkpoint_arg} \
    --encoder_name ${ENCODER_NAME} \
    --num_steps ${NUM_STEPS} \
    --cfg_guidance_scale ${CFG_GUIDANCE_SCALE} \
    --seed ${SEED} \
    --post_process ${POST_PROCESS} \
    ${label_noise_config_arg} \
    --guidance_region ${GUIDANCE_REGION} \
    --classifier_guidance_schedule ${CLASSIFIER_GUIDANCE_SCHEDULE} \
    --mask_erosion_kernel ${MASK_EROSION_KERNEL} \
    --generation_image_size ${GENERATION_IMAGE_SIZE} \
    ${controlnet_args}"

srun --ntasks=1 --kill-on-bad-exit=1 ./SLURM/run_command.sh "$run_command" 16g
