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
module load python/3.10
module load cuda

source .venv/bin/activate

# Move to project directory
cd /workdir/martinije

# Run script
python notebooks/augmented_training.py