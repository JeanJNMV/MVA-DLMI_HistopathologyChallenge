# %%
import argparse
from pathlib import Path
import sys

import torch
import torchmetrics
import warnings
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlmi.utils import set_seed, get_device
from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.transforms import get_default_img_size, get_ood_transform
from dlmi.train import train
from dlmi.test import evaluate_no_tta, evaluate_with_tta

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
parser.add_argument("--train_path", type=str, default="data/train.h5")
parser.add_argument("--val_path", type=str, default="data/val.h5")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--figures_dir", type=str, default="figures")
args = parser.parse_args()

MODEL_NAME = args.model_name
NUM_UNFREEZE = args.num_unfreeze
TRAIN_PATH = Path(args.train_path)
VAL_PATH = Path(args.val_path)
MODEL_DIR = Path(args.model_dir)
FIGURES_DIR = Path(args.figures_dir)
MODEL_SAVE_PATH = MODEL_DIR / f"augmented_{MODEL_NAME}_{NUM_UNFREEZE}_layers.pth"
CURVES_SAVE_PATH = FIGURES_DIR / f"training_curves_{MODEL_NAME}_{NUM_UNFREEZE}_layers.png"

# Hyperparameters
SEED = 0
BATCH_SIZE = 16
LR = 0.001
NUM_EPOCHS = 100
PATIENCE = 10
IMG_SIZE = get_default_img_size(MODEL_NAME)

# %%
set_seed(SEED)
device = get_device()
torch.backends.cudnn.benchmark = True
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}, Unfrozen blocks: {NUM_UNFREEZE}")
print(f"Train path: {TRAIN_PATH}")
print(f"Val path: {VAL_PATH}")

# %%
# Data
train_preprocessing = get_ood_transform(size=IMG_SIZE, train=True, model_name=MODEL_NAME)
val_preprocessing = get_ood_transform(size=IMG_SIZE, train=False, model_name=MODEL_NAME)

train_ds = H5Dataset(str(TRAIN_PATH), transform=train_preprocessing, mode="train")
val_ds = H5Dataset(str(VAL_PATH), transform=val_preprocessing, mode="train")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

# %%
# Model
model = get_finetunable_dinov2(
    MODEL_NAME, num_blocks_to_unfreeze=NUM_UNFREEZE, device=device
)

criterion = torch.nn.BCEWithLogitsLoss()
metric = torchmetrics.Accuracy(task="binary")

optimizer = torch.optim.Adam(
    [
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": 1e-5,
        },
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
    save_path=str(MODEL_SAVE_PATH),
)

# %%
# Evaluation
model.load_state_dict(
    torch.load(str(MODEL_SAVE_PATH), weights_only=True, map_location=device)
)

acc = evaluate_no_tta(model, val_loader, device)
print(f"Val Accuracy (no TTA): {acc:.4f}")

# %%
# TTA
acc = evaluate_with_tta(model, str(VAL_PATH), val_preprocessing, device)
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
plt.savefig(str(CURVES_SAVE_PATH))

print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"Training curves saved to: {CURVES_SAVE_PATH}")
