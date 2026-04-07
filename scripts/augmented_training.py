import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import torch
import torchmetrics
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

SEED = 0
BATCH_SIZE = 16
LR = 0.001
NUM_EPOCHS = 100
PATIENCE = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--num_unfreeze", type=int, default=2)
    parser.add_argument("--train_path", type=str, default="data/train.h5")
    parser.add_argument("--val_path", type=str, default="data/val.h5")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--figures_dir", type=str, default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    num_unfreeze = args.num_unfreeze
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    model_dir = Path(args.model_dir)
    figures_dir = Path(args.figures_dir)
    model_save_path = model_dir / f"augmented_{model_name}_{num_unfreeze}_layers.pth"
    curves_save_path = figures_dir / f"training_curves_{model_name}_{num_unfreeze}_layers.png"
    img_size = get_default_img_size(model_name)

    set_seed(SEED)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {model_name}, Unfrozen blocks: {num_unfreeze}")
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")

    train_preprocessing = get_ood_transform(size=img_size, train=True, model_name=model_name)
    val_preprocessing = get_ood_transform(size=img_size, train=False, model_name=model_name)

    train_ds = H5Dataset(str(train_path), transform=train_preprocessing, mode="train")
    val_ds = H5Dataset(str(val_path), transform=val_preprocessing, mode="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    model = get_finetunable_dinov2(
        model_name, num_blocks_to_unfreeze=num_unfreeze, device=device
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
        save_path=str(model_save_path),
    )

    model.load_state_dict(
        torch.load(str(model_save_path), weights_only=True, map_location=device)
    )

    acc = evaluate_no_tta(model, val_loader, device)
    print(f"Val Accuracy (no TTA): {acc:.4f}")

    acc = evaluate_with_tta(model, str(val_path), val_preprocessing, device)
    print(f"Val Accuracy with TTA: {acc:.4f}")

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
    fig.tight_layout()
    fig.savefig(str(curves_save_path))

    print(f"Model saved to: {model_save_path}")
    print(f"Training curves saved to: {curves_save_path}")


if __name__ == "__main__":
    main()
