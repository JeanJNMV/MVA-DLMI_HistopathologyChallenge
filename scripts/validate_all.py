# Evaluate all trained DINOv2 checkpoints (all model sizes × layer counts) on the
# validation set, with and without TTA. Saves accuracy results to results/ as JSON.

# %%
import json
import os
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torch.utils.data import DataLoader

from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.test import evaluate_no_tta, evaluate_with_tta
from dlmi.transforms import get_ood_transform
from dlmi.utils import get_device, set_seed

warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## Configuration

# %%
REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = str(REPO_ROOT / "data" / "val.h5")
MODELS_DIR = str(REPO_ROOT / "models")
RESULTS_DIR = str(REPO_ROOT / "results")
IMG_SIZE = 98
BATCH_SIZE = 16
SEED = 0

MODEL_NAMES = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"]
NB_LAYERS_LIST = [2, 3, 5, 7, 10]

set_seed(SEED)
device = get_device()
print(f"Device: {device}")

# %% [markdown]
# ## Prepare validation data

# %%
val_preprocessing = get_ood_transform(size=IMG_SIZE, train=False)
val_ds = H5Dataset(VAL_PATH, transform=val_preprocessing, mode="train")
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"Val: {len(val_ds)} samples")

# %% [markdown]
# ## Run evaluation over all model checkpoints

# %%
results_no_tta = {}
results_tta = {}

for model_name in MODEL_NAMES:
    for nb_layers in NB_LAYERS_LIST:
        key = f"{model_name}_{nb_layers}_layers"
        model_path = os.path.join(MODELS_DIR, f"augmented_{key}.pth")

        if not os.path.exists(model_path):
            print(f"[SKIP] {model_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {key}")
        print(f"{'=' * 60}")

        model = get_finetunable_dinov2(
            model_name, num_blocks_to_unfreeze=nb_layers, device=device
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Without TTA
        acc_no_tta = evaluate_no_tta(model, val_loader, device)
        results_no_tta[key] = acc_no_tta
        print(f"  No TTA accuracy: {acc_no_tta:.4f}")

        # With TTA
        acc_tta = evaluate_with_tta(model, VAL_PATH, val_preprocessing, device)
        results_tta[key] = acc_tta
        print(f"  TTA accuracy:    {acc_tta:.4f}")

print(f"\n\nEvaluated {len(results_no_tta)} models.")

# %% [markdown]
# ## Save results to JSON

# %%
os.makedirs(RESULTS_DIR, exist_ok=True)

no_tta_path = os.path.join(RESULTS_DIR, "val_results_no_tta.json")
tta_path = os.path.join(RESULTS_DIR, "val_results_tta.json")

with open(no_tta_path, "w") as f:
    json.dump(results_no_tta, f, indent=2)

with open(tta_path, "w") as f:
    json.dump(results_tta, f, indent=2)

print(f"Saved: {no_tta_path}")
print(f"Saved: {tta_path}")
