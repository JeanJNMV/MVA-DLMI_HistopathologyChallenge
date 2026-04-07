# Hyperparameter search for DINOv2 fine-tuning using Optuna (TPE sampler).
# Explores model size, number of unfrozen blocks, augmentation, and optimiser
# settings. Saves the best checkpoint and its config to models/.

import argparse
import json
import sys
import warnings
from pathlib import Path

import optuna
import torch
import torchmetrics
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torch.utils.data import DataLoader

from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.train import train
from dlmi.transforms import HEDJitter, StainMix, build_stain_bank, get_ood_transform
from dlmi.utils import get_device, set_seed

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths (relative to the repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = str(REPO_ROOT / "data" / "train.h5")
VAL_PATH = str(REPO_ROOT / "data" / "val.h5")
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = str(MODELS_DIR / "optuna_best.pth")
BEST_CONFIG_PATH = str(MODELS_DIR / "optuna_best_config.json")

PATIENCE = 5
NUM_EPOCHS = 100
SEED = 0

# ---------------------------------------------------------------------------
# Global tracker for the best validation accuracy seen so far
# ---------------------------------------------------------------------------
_best_val_acc = 0.0


def build_transforms(trial, stain_bank):
    """Build train/val transforms from Optuna-sampled augmentation params."""
    img_size = trial.suggest_categorical("img_size", [98, 112, 196, 224])
    hed_theta = trial.suggest_float("hed_theta", 0.01, 0.20)
    use_stain_mix = trial.suggest_categorical("use_stain_mix", [True, False])
    stain_mix_alpha = (
        trial.suggest_float("stain_mix_alpha", 0.1, 0.7) if use_stain_mix else 0.3
    )

    train_augs = [
        T.Resize((img_size, img_size)),
        HEDJitter(theta=hed_theta),
    ]
    if use_stain_mix and stain_bank is not None:
        train_augs.append(StainMix(stain_bank, alpha=stain_mix_alpha))
    train_augs += [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
    ]
    train_transform = T.Compose(train_augs)
    val_transform = T.Resize((img_size, img_size))
    return train_transform, val_transform, img_size


def objective(trial):
    global _best_val_acc

    set_seed(SEED)
    device = get_device()

    # ── Model hyperparameters ──────────────────────────────────────────────
    model_name = trial.suggest_categorical(
        "model_name",
        ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
    )
    nb_layers = trial.suggest_int("nb_layers_to_fine_tune", 2, 10)

    use_mixstyle = trial.suggest_categorical("use_mixstyle", [True, False])
    mixstyle_p = trial.suggest_float("mixstyle_p", 0.1, 0.9) if use_mixstyle else 0.5
    mixstyle_alpha = (
        trial.suggest_float("mixstyle_alpha", 0.05, 0.5) if use_mixstyle else 0.1
    )

    # ── Optimiser hyperparameters ──────────────────────────────────────────
    head_lr = trial.suggest_float("head_lr", 1e-4, 1e-2, log=True)
    backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # ── Stain bank (uses a temporary base dataset at default size) ─────────
    stain_bank_size = trial.suggest_int("stain_bank_size", 100, 1000, step=100)
    base_transform = get_ood_transform(size=98, train=False)
    base_ds = H5Dataset(TRAIN_PATH, transform=base_transform, mode="train")
    stain_bank = build_stain_bank(base_ds, max_images=stain_bank_size)

    # ── Augmentation transforms ────────────────────────────────────────────
    train_transform, val_transform, img_size = build_transforms(trial, stain_bank)

    train_ds = H5Dataset(TRAIN_PATH, transform=train_transform, mode="train")
    val_ds = H5Dataset(VAL_PATH, transform=val_transform, mode="train")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────
    model = get_finetunable_dinov2(
        model_name,
        num_blocks_to_unfreeze=nb_layers,
        device=device,
        use_mixstyle=use_mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
    )
    criterion = torch.nn.BCELoss()
    metric = torchmetrics.Accuracy("binary")
    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    )

    # ── Training ───────────────────────────────────────────────────────────
    # Use a per-trial temp path; overwrite with best only if it wins
    trial_save_path = str(MODELS_DIR / f"optuna_trial_{trial.number}.pth")

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
        save_path=trial_save_path,
    )

    # Best val accuracy achieved during this trial
    best_val_acc_trial = max(history["val_metric"])

    # ── Track global best and persist ─────────────────────────────────────
    if best_val_acc_trial > _best_val_acc:
        _best_val_acc = best_val_acc_trial

        # Copy weights to the canonical best-model path
        import shutil

        shutil.copy2(trial_save_path, BEST_MODEL_PATH)

        config = {
            "trial_number": trial.number,
            "val_accuracy": float(best_val_acc_trial),
            "model_name": model_name,
            "nb_layers_to_fine_tune": nb_layers,
            "use_mixstyle": use_mixstyle,
            "mixstyle_p": float(mixstyle_p),
            "mixstyle_alpha": float(mixstyle_alpha),
            "head_lr": float(head_lr),
            "backbone_lr": float(backbone_lr),
            "batch_size": batch_size,
            "img_size": img_size,
            "hed_theta": float(trial.params.get("hed_theta", 0.05)),
            "use_stain_mix": bool(trial.params.get("use_stain_mix", False)),
            "stain_mix_alpha": float(trial.params.get("stain_mix_alpha", 0.3)),
            "stain_bank_size": stain_bank_size,
        }
        with open(BEST_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

        print("\n" + "=" * 60)
        print(
            f"  NEW BEST — Trial {trial.number}  |  Val Acc: {best_val_acc_trial:.4f}"
        )
        print("  Configuration:")
        for k, v in config.items():
            print(f"    {k}: {v}")
        print("=" * 60 + "\n")

    # Clean up trial-specific file to avoid filling the disk
    Path(trial_save_path).unlink(missing_ok=True)

    return best_val_acc_trial


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna HP search for DINOv2 fine-tuning"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--study_name", type=str, default="dinov2_hps", help="Optuna study name"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///optuna.db). "
        "Omit to use in-memory storage.",
    )
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    print(f"Starting Optuna search: {args.n_trials} trials, study='{args.study_name}'")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n===== BEST TRIAL =====")
    best = study.best_trial
    print(f"  Value (val acc): {best.value:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"\nBest model saved to: {BEST_MODEL_PATH}")
    print(f"Best config saved to: {BEST_CONFIG_PATH}")
