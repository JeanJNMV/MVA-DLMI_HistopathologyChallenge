#!/bin/bash
#SBATCH --job-name=dinov2_train
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load environment
module load miniconda3/25.5.1/none-none

source .venv/bin/activate

# Move to project directory
cd /workdir/martinije/dlmi_challenge

# Run script
python scripts/augmented_training.py 
python scripts/augmented_training.py --num-unfreeze 3
python scripts/augmented_training.py --num-unfreeze 5
python scripts/augmented_training.py --model-name dinov2_vitb14
python scripts/augmented_training.py --model-name dinov2_vitb14 --num-unfreeze 3
python scripts/augmented_training.py --model-name dinov2_vitb14 --num-unfreeze 5