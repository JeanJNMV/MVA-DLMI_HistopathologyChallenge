#!/bin/bash
#SBATCH --job-name=final
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.err

# Usage (set hyperparams from LOCO results):
#   sbatch --export=MODEL_NAME=hibou-l,NUM_UNFREEZE=2,NUM_EPOCHS=40 scripts/final_training.slurm

set -euo pipefail

# --- Ruche modules ---
module purge
PYTHON_MODULE="${PYTHON_MODULE:-python/3.14.0/gcc-15.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0.2/none-none}"
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"

# --- Paths ---
WORK_ROOT="${WORKDIR:-/workdir/$USER}"
PROJECT_DIR="${PROJECT_DIR:-$WORK_ROOT/MVA-DLMI_Histopathology_Challenge}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK_ROOT/.cache/torch}"

# --- Config (from LOCO results) ---
MODEL_NAME="${MODEL_NAME:-hibou-l}"
NUM_UNFREEZE="${NUM_UNFREEZE:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
TRAIN_PATH="${TRAIN_PATH:-$PROJECT_DIR/data/train.h5}"
VAL_PATH="${VAL_PATH:-$PROJECT_DIR/data/val.h5}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
FIGURES_DIR="${FIGURES_DIR:-$PROJECT_DIR/figures}"

BATCH_SIZE="${BATCH_SIZE:-64}"
LR_HEAD="${LR_HEAD:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
IMG_SIZE="${IMG_SIZE:-}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-$SLURM_CPUS_PER_TASK}"

cd "$PROJECT_DIR"
mkdir -p logs "$MODEL_DIR" "$FIGURES_DIR" "$HF_HOME" "$TORCH_HOME"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "========================================"
echo "Final training on all 4 centers"
echo "Model: $MODEL_NAME  |  Unfreeze: $NUM_UNFREEZE  |  Epochs: $NUM_EPOCHS"
echo "Batch: $BATCH_SIZE  |  Workers: $NUM_WORKERS"
echo "Image size: ${IMG_SIZE:-auto}"
echo "Modules: $PYTHON_MODULE | $CUDA_MODULE"
echo "HF_HOME: $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"
echo "========================================"

img_size_args=()
if [ -n "$IMG_SIZE" ]; then
    img_size_args=(--img_size "$IMG_SIZE")
fi

"$PYTHON_BIN" scripts/final_training.py \
    --model_name "$MODEL_NAME" \
    --num_unfreeze "$NUM_UNFREEZE" \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --model_dir "$MODEL_DIR" \
    --figures_dir "$FIGURES_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr_head "$LR_HEAD" \
    --lr_backbone "$LR_BACKBONE" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    "${img_size_args[@]}"
