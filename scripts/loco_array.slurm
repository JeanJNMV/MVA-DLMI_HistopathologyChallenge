#!/bin/bash
#SBATCH --job-name=loco
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=logs/loco_%A_%a.out
#SBATCH --error=logs/loco_%A_%a.err

set -euo pipefail

module purge
PYTHON_MODULE="${PYTHON_MODULE:-python/3.14.0/gcc-15.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0.2/none-none}"
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"

WORK_ROOT="${WORKDIR:-/workdir/$USER}"
PROJECT_DIR="${PROJECT_DIR:-$WORK_ROOT/MVA-DLMI_Histopathology_Challenge}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK_ROOT/.cache/torch}"

MODEL_NAME="${MODEL_NAME:-dinov2_vitl14}"
NUM_UNFREEZE="${NUM_UNFREEZE:-5}"
TRAIN_PATH="${TRAIN_PATH:-$PROJECT_DIR/data/train.h5}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_DIR/models/loco}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results}"

BATCH_SIZE="${BATCH_SIZE:-64}"
LR_HEAD="${LR_HEAD:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
IMG_SIZE="${IMG_SIZE:-}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-$SLURM_CPUS_PER_TASK}"

CENTERS=(0 3 4)
HELD_OUT_CENTER="${CENTERS[$SLURM_ARRAY_TASK_ID]}"

cd "$PROJECT_DIR"
mkdir -p logs "$SAVE_DIR" "$RESULTS_DIR" "$HF_HOME" "$TORCH_HOME"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    echo "Run 'uv sync' in $PROJECT_DIR on the login node first." >&2
    exit 1
fi

echo "========================================"
echo "LOCO fold: held-out center $HELD_OUT_CENTER"
echo "Model: $MODEL_NAME  |  Unfreeze: $NUM_UNFREEZE"
echo "Batch size: $BATCH_SIZE  |  Workers: $NUM_WORKERS"
echo "Image size: ${IMG_SIZE:-auto}"
echo "Modules: $PYTHON_MODULE | $CUDA_MODULE"
echo "HF_HOME: $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"
echo "========================================"

img_size_args=()
if [ -n "$IMG_SIZE" ]; then
    img_size_args=(--img_size "$IMG_SIZE")
fi

"$PYTHON_BIN" scripts/loco_cv.py \
    --model_name "$MODEL_NAME" \
    --num_unfreeze "$NUM_UNFREEZE" \
    --train_path "$TRAIN_PATH" \
    --batch_size "$BATCH_SIZE" \
    --lr_head "$LR_HEAD" \
    --lr_backbone "$LR_BACKBONE" \
    --num_epochs "$NUM_EPOCHS" \
    --patience "$PATIENCE" \
    --save_dir "$SAVE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --held_out_center "$HELD_OUT_CENTER" \
    "${img_size_args[@]}"
