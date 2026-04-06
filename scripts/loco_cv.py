"""Leave-One-Center-Out (LOCO) cross-validation for hyperparameter selection.

For each fold, one of the 3 training centers (0, 3, 4) is held out as a
simulated OOD validation set, and the model is trained on the remaining two.
This gives an unbiased estimate of generalization to unseen centers, avoiding
the optimistic bias that comes from tuning hyperparameters directly on the
official validation set (center 1).

Usage
-----
    python scripts/loco_cv.py --model_name dinov2_vits14 --num_unfreeze 2

The script reports per-fold accuracy and the mean LOCO accuracy.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torchmetrics
import warnings

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlmi.utils import set_seed, get_device
from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.transforms import get_default_img_size, get_ood_transform
from dlmi.train import train
from dlmi.test import evaluate_no_tta

warnings.filterwarnings("ignore", category=UserWarning)

# Training centers available in train.h5
ALL_TRAIN_CENTERS = [0, 3, 4]


def run_one_fold(
    fold_idx,
    held_out_center,
    train_centers,
    train_path,
    model_name,
    num_unfreeze,
    device,
    batch_size=16,
    lr_head=1e-3,
    lr_backbone=1e-5,
    num_epochs=100,
    patience=10,
    img_size=98,
    save_dir=None,
    num_workers=0,
):
    """Train on `train_centers` and evaluate on `held_out_center`.

    Parameters
    ----------
    fold_idx : int
        Fold number (for logging/saving).
    held_out_center : int
        Center ID used as simulated OOD validation.
    train_centers : list of int
        Center IDs used for training.
    train_path : str
        Path to the training HDF5 file.
    model_name : str
        DINOv2 variant name.
    num_unfreeze : int
        Number of transformer blocks to unfreeze.
    device : torch.device
        Compute device.
    batch_size : int
        Batch size.
    lr_head : float
        Learning rate for the classification head.
    lr_backbone : float
        Learning rate for unfrozen backbone layers.
    num_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    img_size : int
        Input image size.
    save_dir : str or None
        Directory to save fold model checkpoints.

    Returns
    -------
    float
        Validation accuracy on the held-out center.
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: train on centers {train_centers}, validate on center {held_out_center}")
    print(f"{'='*60}")

    train_transform = get_ood_transform(size=img_size, train=True, model_name=model_name)
    val_transform = get_ood_transform(size=img_size, train=False, model_name=model_name)

    train_ds = H5Dataset(train_path, transform=train_transform, mode="train", centers=train_centers)
    val_ds = H5Dataset(train_path, transform=val_transform, mode="train", centers=[held_out_center])

    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Fresh model for each fold
    model = get_finetunable_dinov2(model_name, num_blocks_to_unfreeze=num_unfreeze, device=device)

    criterion = torch.nn.BCEWithLogitsLoss()
    metric = torchmetrics.Accuracy(task="binary")

    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": lr_backbone,
            },
            {"params": model.head.parameters(), "lr": lr_head},
        ]
    )

    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"loco_center{held_out_center}_{model_name}_{num_unfreeze}layers.pth",
        )

    history = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        metric,
        device,
        num_epochs=num_epochs,
        patience=patience,
        save_path=save_path,
    )

    # Reload best checkpoint for evaluation
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))

    acc = evaluate_no_tta(model, val_loader, device)
    print(f"\n  Fold {fold_idx} result: center {held_out_center} accuracy = {acc:.4f}")

    return acc


def main():
    parser = argparse.ArgumentParser(description="LOCO cross-validation")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--num_unfreeze", type=int, default=2)
    parser.add_argument("--train_path", type=str, default="data/train.h5")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="models/loco")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--held_out_center", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True

    if args.held_out_center is not None and args.held_out_center not in ALL_TRAIN_CENTERS:
        raise ValueError(
            f"held_out_center must be one of {ALL_TRAIN_CENTERS}, got {args.held_out_center}."
        )

    centers_to_run = (
        [args.held_out_center]
        if args.held_out_center is not None
        else ALL_TRAIN_CENTERS
    )

    print(f"Device: {device}")
    print(f"Model: {args.model_name}, Unfrozen blocks: {args.num_unfreeze}")
    print(f"LOCO CV over centers: {centers_to_run}")
    print(f"DataLoader workers: {args.num_workers}")

    img_size = (
        args.img_size
        if args.img_size is not None
        else get_default_img_size(args.model_name)
    )
    print(f"Image size: {img_size}")

    fold_results = {}
    for fold_idx, held_out in enumerate(centers_to_run):
        train_centers = [c for c in ALL_TRAIN_CENTERS if c != held_out]
        acc = run_one_fold(
            fold_idx=fold_idx,
            held_out_center=held_out,
            train_centers=train_centers,
            train_path=args.train_path,
            model_name=args.model_name,
            num_unfreeze=args.num_unfreeze,
            device=device,
            batch_size=args.batch_size,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            num_epochs=args.num_epochs,
            patience=args.patience,
            img_size=img_size,
            save_dir=args.save_dir,
            num_workers=args.num_workers,
        )
        fold_results[f"center_{held_out}"] = acc

    print(f"\n{'='*60}")
    print("LOCO Results")
    print(f"{'='*60}")
    for center in centers_to_run:
        print(f"  Held-out center {center}: {fold_results[f'center_{center}']:.4f}")
    if len(fold_results) == len(ALL_TRAIN_CENTERS):
        mean_acc = sum(fold_results.values()) / len(fold_results)
        print(f"  Mean LOCO accuracy:    {mean_acc:.4f}")
    print(f"{'='*60}")

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    suffix = (
        f"_center{args.held_out_center}"
        if args.held_out_center is not None
        else ""
    )
    results_path = os.path.join(
        args.results_dir,
        f"loco_{args.model_name}_{args.num_unfreeze}layers{suffix}.json",
    )
    results_payload = {
        "model_name": args.model_name,
        "num_unfreeze": args.num_unfreeze,
        "fold_results": fold_results,
    }
    if args.held_out_center is None:
        results_payload["mean_loco_accuracy"] = (
            sum(fold_results.values()) / len(fold_results)
        )
    else:
        results_payload["held_out_center"] = args.held_out_center

    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
