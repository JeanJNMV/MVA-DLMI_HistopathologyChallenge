# Train a DINOv2 model with augmented data (HED jitter, flips, rotations) on the
# histopathology dataset. Accepts --model_name and --num_unfreeze CLI arguments.
# Saves the best model checkpoint and training-curve plots to models/ and results/.

# %%
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchmetrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torch.utils.data import DataLoader

from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.test import evaluate_no_tta, evaluate_with_tta
from dlmi.train import train
from dlmi.transforms import build_stain_bank, get_ood_transform
from dlmi.utils import get_device, set_seed

warnings.filterwarnings("ignore", category=UserWarning)

# %%
# ----------------------
# Argument parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="dinov2_vits14",
    help="Model name (e.g., dinov2_vits14, dinov2_vitb14, ...)",
)
parser.add_argument(
    "--num_unfreeze",
    type=int,
    default=2,
    help="Number of transformer blocks to unfreeze",
)
args = parser.parse_args()

MODEL_NAME = args.model_name
NUM_UNFREEZE = args.num_unfreeze

# %%
# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = str(REPO_ROOT / "data" / "train.h5")
VAL_PATH = str(REPO_ROOT / "data" / "val.h5")
TEST_PATH = str(REPO_ROOT / "data" / "test.h5")

MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = str(MODELS_DIR / f"augmented_{MODEL_NAME}_{NUM_UNFREEZE}_layers.pth")

CURVES_DIR = REPO_ROOT / "results" / "training_curves"
CURVES_DIR.mkdir(parents=True, exist_ok=True)
CURVES_SAVE_PATH = str(
    CURVES_DIR / f"training_curves_{MODEL_NAME}_{NUM_UNFREEZE}_layers.png"
)

# Hyperparameters
SEED = 0
BATCH_SIZE = 16
LR = 0.001
NUM_EPOCHS = 100
PATIENCE = 10
IMG_SIZE = 98

USE_MIXSTYLE = True
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1
STAIN_BANK_SIZE = 500

# %%
set_seed(SEED)
device = get_device()
torch.backends.cudnn.benchmark = True
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}, Unfrozen blocks: {NUM_UNFREEZE}")

# %%
# Data
base_transform = get_ood_transform(size=IMG_SIZE, train=False)
base_ds = H5Dataset(TRAIN_PATH, transform=base_transform, mode="train")
stain_bank = build_stain_bank(base_ds, max_images=STAIN_BANK_SIZE)

train_preprocessing = get_ood_transform(
    size=IMG_SIZE, train=True, stain_bank=stain_bank
)
val_preprocessing = get_ood_transform(size=IMG_SIZE, train=False)

train_ds = H5Dataset(TRAIN_PATH, transform=train_preprocessing, mode="train")
val_ds = H5Dataset(VAL_PATH, transform=val_preprocessing, mode="train")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

# %%
# Model
model = get_finetunable_dinov2(
    MODEL_NAME,
    num_blocks_to_unfreeze=NUM_UNFREEZE,
    device=device,
    use_mixstyle=USE_MIXSTYLE,
    mixstyle_p=MIXSTYLE_P,
    mixstyle_alpha=MIXSTYLE_ALPHA,
)

criterion = torch.nn.BCELoss()
metric = torchmetrics.Accuracy(task="binary")

optimizer = torch.optim.Adam(
    [
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(), "lr": LR},
    ]
)

# %%
# Training
history = train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    metric,
    device,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    save_path=MODEL_SAVE_PATH,
)

# %%
# Evaluation
model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))

acc = evaluate_no_tta(model, val_loader, device)
print(f"Val Accuracy (no TTA): {acc:.4f}")

# %%
# TTA
acc = evaluate_with_tta(model, VAL_PATH, val_preprocessing, device)
print(f"Val Accuracy with TTA: {acc:.4f}")

# %%
# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history["train_metric"], label="Train")
axes[1].plot(history["val_metric"], label="Val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(CURVES_SAVE_PATH)

print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"Training curves saved to: {CURVES_SAVE_PATH}")
