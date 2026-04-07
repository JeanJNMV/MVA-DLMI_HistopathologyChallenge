# Histopathology OOD Classification

Our code for the Kaggle challenge **[MVA DLMI 2026 — Histopathology OOD Classification](https://www.kaggle.com/competitions/mva-dlmi-2026-histopathology-ood-classification)**, the final assignment for the "Deep Learning for Medical Imaging" course at MVA (ENS Paris-Saclay) and CentraleSupélec.

The task is binary classification of whole-slide-image patches under significant **distribution shift**: training data originates from 3 hospital centers while validation and test sets come from unseen centers with different staining procedures, equipment, and conditions.

## Key Features
- Fine-tuned [DINOv2](https://github.com/facebookresearch/dinov2) backbones (ViT-S/B/L-14) with partial unfreezing
- Used [Hibou](https://huggingface.co/histai), a pathology-specialized vision transformers (Hibou-B/L)
- **MixStyle** domain-generalization regularization applied in token space
- **HED stain jitter** and **StainMix** augmentations to simulate inter-center stain variability
- **Test-Time Augmentation (TTA)** via geometric transforms (flips + rotations)
- **Leave-One-Center-Out (LOCO)** cross-validation for robust model selection
- Hyperparameter search with **Optuna** (TPE sampler)
- Final training on combined train+val data with cosine annealing & mixed-precision

## Project Structure

```
.
├── data/                        # HDF5 datasets (train / val / test)
├── models/                      # Saved model checkpoints (.pth)
├── notebooks/
│   ├── 0. Getting started.ipynb
│   ├── 1. Visualization.ipynb
│   ├── 2. Baseline training.ipynb
│   ├── 3. Augmented training.ipynb
│   └── 4. Model evaluation.ipynb
├── results/
│   ├── training_curves/         # Loss & accuracy plots per checkpoint
│   ├── val_results_no_tta.json  # Validation accuracy without TTA
│   ├── val_results_tta.json     # Validation accuracy with TTA
│   └── *.csv                    # Kaggle submission files
├── report/                      # LaTeX report & bibliography
├── scripts/
│   ├── augmented_training.py    # Train a single model checkpoint
│   ├── final_training.py        # Train on combined train+val (all centers)
│   ├── loco_cv.py               # Leave-One-Center-Out cross-validation
│   ├── aggregate_loco_results.py# Aggregate LOCO fold results into summary
│   ├── optuna_search.py         # Hyperparameter search with Optuna
│   ├── validate_all.py          # Evaluate all checkpoints on the val set
│   ├── generate_submissions.py  # Generate Kaggle CSV files
│   ├── smoke_test.py            # Verify install, GPU, data & model loading
│   ├── final_training.slurm     # SLURM job for final training (gpua100)
│   ├── loco_array.slurm         # SLURM array job for parallel LOCO CV
│   └── smoke_test.slurm         # SLURM quick smoke-test job
└── src/dlmi/
    ├── dataset.py               # H5Dataset & PrecomputedDataset
    ├── model.py                 # DINOv2 backbones + MixStyle + linear head
    ├── train.py                 # Training loop with early stopping
    ├── test.py                  # TTA inference & evaluation helpers
    ├── transforms.py            # HEDJitter, StainMix, OOD transform pipeline
    └── utils.py                 # Seed, device, submission helpers
```

## Getting Started

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.10 (CUDA, MPS, or CPU)

### Installation

```bash
git clone https://github.com/JeanJNMV/MVA-DLMI_HistopathologyChallenge.git
cd MVA-DLMI_Histopathology_Challenge

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package and its dependencies
pip install -e .
```

Alternatively, you can use `uv sync` in your terminal if you have `uv` installed.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 0 | [Getting started](notebooks/0.%20Getting%20started.ipynb) | Dataset exploration provided by challenge organizers |
| 1 | [Visualization](notebooks/1.%20Visualization.ipynb) | Visual inspection of patches across centers |
| 2 | [Baseline training](notebooks/2.%20Baseline%20training.ipynb) | DINOv2 feature extractor + frozen linear probe |
| 3 | [Augmented training](notebooks/3.%20Augmented%20training.ipynb) | Fine-tuning with MixStyle, HED jitter, StainMix |
| 4 | [Model evaluation](notebooks/4.%20Model%20evaluation.ipynb) | Comparison of all checkpoints with/without TTA |

## Authors

- **Jean-Vincent Martini** — MVA, ENS Paris-Saclay & CentraleSupélec
- **Ayoub El Kbadi** — MVA, ENS Paris-Saclay & CentraleSupélec