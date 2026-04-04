#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

# Step 0: verify the environment before any full training.
# Usage:
#   sbatch scripts/smoke_test.slurm                      # quick check on gpu_test (V100)
#   sbatch --export=MODEL_NAME=hibou-l scripts/smoke_test.slurm  # also test HuggingFace download

set -euo pipefail

module purge
module load cuda/12.1  # adjust as needed

WORK_ROOT="${WORKDIR:-/workdir/$USER}"
PROJECT_DIR="${PROJECT_DIR:-$WORK_ROOT/MVA-DLMI_Histopathology_Challenge}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK_ROOT/.cache/torch}"

MODEL_NAME="${MODEL_NAME:-dinov2_vits14}"
TRAIN_PATH="${TRAIN_PATH:-$PROJECT_DIR/data/train.h5}"
VAL_PATH="${VAL_PATH:-$PROJECT_DIR/data/val.h5}"

cd "$PROJECT_DIR"
mkdir -p logs "$HF_HOME" "$TORCH_HOME"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found: $PYTHON_BIN" >&2
    echo "Run 'uv sync' in $PROJECT_DIR on the login node first." >&2
    exit 1
fi

echo "Project : $PROJECT_DIR"
echo "Python  : $PYTHON_BIN"
echo "Model   : $MODEL_NAME"

"$PYTHON_BIN" scripts/smoke_test.py \
    --model_name "$MODEL_NAME" \
    --train_path "$TRAIN_PATH" \
    --val_path   "$VAL_PATH"
