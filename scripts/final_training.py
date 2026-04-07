import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import torch
import torchmetrics
from torch.utils.data import ConcatDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlmi.utils import set_seed, get_device
from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.transforms import get_default_img_size, get_ood_transform
from dlmi.train import train_one_epoch

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Final training on all 4 centers")
    parser.add_argument("--model_name", type=str, default="hibou-l")
    parser.add_argument("--num_unfreeze", type=int, default=2)
    parser.add_argument("--train_path", type=str, default="data/train.h5")
    parser.add_argument("--val_path", type=str, default="data/val.h5")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    num_unfreeze = args.num_unfreeze
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    model_dir = Path(args.model_dir)
    figures_dir = Path(args.figures_dir)
    model_save_path = model_dir / f"final_{model_name}_{num_unfreeze}_layers.pth"
    curves_save_path = figures_dir / f"final_curves_{model_name}_{num_unfreeze}_layers.png"

    set_seed(args.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {model_name}  |  Unfreeze: {num_unfreeze}")
    print(f"Epochs: {args.num_epochs}  |  Batch: {args.batch_size}")

    img_size = args.img_size or get_default_img_size(model_name)
    print(f"Image size: {img_size}")

    transform = get_ood_transform(size=img_size, train=True, model_name=model_name)
    train_ds = H5Dataset(str(train_path), transform=transform, mode="train")
    val_ds = H5Dataset(str(val_path), transform=transform, mode="train")
    all_ds = ConcatDataset([train_ds, val_ds])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    all_loader = DataLoader(all_ds, shuffle=True, **loader_kwargs)
    print(f"Total samples: {len(all_ds)}  ({len(train_ds)} train + {len(val_ds)} val)")

    model = get_finetunable_dinov2(
        model_name, num_blocks_to_unfreeze=num_unfreeze, device=device
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    metric = torchmetrics.Accuracy(task="binary")
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": args.lr_backbone,
            },
            {"params": model.head.parameters(), "lr": args.lr_head},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-7
    )
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None
    if scaler:
        print("AMP enabled")

    history = {"train_loss": [], "train_metric": []}

    for epoch in range(args.num_epochs):
        loss, acc = train_one_epoch(
            model, all_loader, optimizer, criterion, metric, device, scaler=scaler
        )
        scheduler.step()
        history["train_loss"].append(loss)
        history["train_metric"].append(acc)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] | Loss {loss:.4f} | Acc {acc:.4f}")

    torch.save(model.state_dict(), str(model_save_path))
    print(f"\nModel saved to: {model_save_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(history["train_metric"])
    axes[1].set_title("Train Accuracy")
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(str(curves_save_path))
    print(f"Curves saved to: {curves_save_path}")


if __name__ == "__main__":
    main()
