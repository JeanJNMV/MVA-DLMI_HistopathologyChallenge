"""Final training on all 4 centers for Kaggle submission.

Trains on train.h5 (centers 0, 3, 4) + val.h5 (center 1) with hyperparameters
selected via LOCO cross-validation. No validation set, no early stopping —
training runs for a fixed number of epochs with cosine annealing.

Usage
-----
    python scripts/final_training.py --model_name hibou-l --num_unfreeze 2 --num_epochs 40
"""

import argparse
from pathlib import Path
import sys

import torch
import torchmetrics
import warnings
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlmi.utils import set_seed, get_device
from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.transforms import get_ood_transform
from dlmi.train import train_one_epoch

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------
# Argument parsing
# ----------------------
parser = argparse.ArgumentParser(description="Final training on all 4 centers")
parser.add_argument("--model_name", type=str, default="hibou-l")
parser.add_argument("--num_unfreeze", type=int, default=2)
parser.add_argument("--train_path", type=str, default="data/train.h5")
parser.add_argument("--val_path", type=str, default="data/val.h5")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--figures_dir", type=str, default="figures")
parser.add_argument("--num_epochs", type=int, default=40,
                    help="Fixed number of epochs (set from LOCO best epoch)")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr_head", type=float, default=1e-3)
parser.add_argument("--lr_backbone", type=float, default=1e-5)
parser.add_argument("--img_size", type=int, default=98)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

MODEL_NAME = args.model_name
NUM_UNFREEZE = args.num_unfreeze
TRAIN_PATH = Path(args.train_path)
VAL_PATH = Path(args.val_path)
MODEL_DIR = Path(args.model_dir)
FIGURES_DIR = Path(args.figures_dir)
MODEL_SAVE_PATH = MODEL_DIR / f"final_{MODEL_NAME}_{NUM_UNFREEZE}_layers.pth"
CURVES_SAVE_PATH = FIGURES_DIR / f"final_curves_{MODEL_NAME}_{NUM_UNFREEZE}_layers.png"

set_seed(args.seed)
device = get_device()
torch.backends.cudnn.benchmark = True
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}  |  Unfreeze: {NUM_UNFREEZE}")
print(f"Epochs: {args.num_epochs}  |  Batch: {args.batch_size}")

# ----------------------
# Data — all 4 centers
# ----------------------
transform = get_ood_transform(size=args.img_size, train=True)

train_ds = H5Dataset(str(TRAIN_PATH), transform=transform, mode="train")  # centers 0, 3, 4
val_ds = H5Dataset(str(VAL_PATH), transform=transform, mode="train")      # center 1
all_ds = ConcatDataset([train_ds, val_ds])

loader_kwargs = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(args.num_workers > 0),
)
all_loader = DataLoader(all_ds, shuffle=True, **loader_kwargs)
print(f"Total samples: {len(all_ds)}  ({len(train_ds)} train + {len(val_ds)} val)")

# ----------------------
# Model
# ----------------------
model = get_finetunable_dinov2(MODEL_NAME, num_blocks_to_unfreeze=NUM_UNFREEZE, device=device)

criterion = torch.nn.BCEWithLogitsLoss()
metric = torchmetrics.Accuracy(task="binary")
optimizer = torch.optim.Adam([
    {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr_backbone},
    {"params": model.head.parameters(), "lr": args.lr_head},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.num_epochs, eta_min=1e-7
)
scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None
if scaler:
    print("AMP enabled")

# ----------------------
# Training loop (no early stopping, no validation)
# ----------------------
history = {"train_loss": [], "train_metric": []}

for epoch in range(args.num_epochs):
    loss, acc = train_one_epoch(
        model, all_loader, optimizer, criterion, metric, device, scaler=scaler
    )
    scheduler.step()
    history["train_loss"].append(loss)
    history["train_metric"].append(acc)
    print(f"Epoch [{epoch + 1}/{args.num_epochs}] | Loss {loss:.4f} | Acc {acc:.4f}")

torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# ----------------------
# Training curves
# ----------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"])
axes[0].set_title("Train Loss")
axes[0].set_xlabel("Epoch")
axes[1].plot(history["train_metric"])
axes[1].set_title("Train Accuracy")
axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(str(CURVES_SAVE_PATH))
print(f"Curves saved to: {CURVES_SAVE_PATH}")
