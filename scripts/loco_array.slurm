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

# Usage:
#   sbatch scripts/loco_array.slurm                          # dinov2_vitl14, 5 layers (defaults)
#   sbatch --export=MODEL_NAME=hibou-l,NUM_UNFREEZE=2 scripts/loco_array.slurm
#   sbatch --export=MODEL_NAME=dinov2_vitl14,NUM_UNFREEZE=3 scripts/loco_array.slurm

set -euo pipefail

# --- Ruche modules (adapt to your environment) ---
module purge
module load cuda/12.1  # adjust version as needed

# --- Paths ---
WORK_ROOT="${WORKDIR:-/workdir/$USER}"
PROJECT_DIR="${PROJECT_DIR:-$WORK_ROOT/MVA-DLMI_Histopathology_Challenge}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

# HuggingFace model cache (used by Hibou-L via timm).
# Point to a shared directory so the model is downloaded only once.
export HF_HOME="${HF_HOME:-$WORK_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK_ROOT/.cache/torch}"

# --- Experiment config (override via sbatch --export=KEY=VALUE) ---
MODEL_NAME="${MODEL_NAME:-dinov2_vitl14}"
NUM_UNFREEZE="${NUM_UNFREEZE:-5}"
TRAIN_PATH="${TRAIN_PATH:-$PROJECT_DIR/data/train.h5}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_DIR/models/loco}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results}"

# A100 has enough VRAM for batch 64 with AMP — 4x fewer steps than batch 16.
BATCH_SIZE="${BATCH_SIZE:-64}"
LR_HEAD="${LR_HEAD:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
IMG_SIZE="${IMG_SIZE:-98}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-$SLURM_CPUS_PER_TASK}"

# --- LOCO fold selection ---
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
echo "HF_HOME: $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"
echo "========================================"

"$PYTHON_BIN" scripts/loco_cv.py \
    --model_name "$MODEL_NAME" \
    --num_unfreeze "$NUM_UNFREEZE" \
    --train_path "$TRAIN_PATH" \
    --batch_size "$BATCH_SIZE" \
    --lr_head "$LR_HEAD" \
    --lr_backbone "$LR_BACKBONE" \
    --num_epochs "$NUM_EPOCHS" \
    --patience "$PATIENCE" \
    --img_size "$IMG_SIZE" \
    --save_dir "$SAVE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --held_out_center "$HELD_OUT_CENTER"
