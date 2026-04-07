import argparse
import json
import os
from pathlib import Path
import sys
import warnings

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.test import evaluate_no_tta, evaluate_with_tta
from dlmi.transforms import get_ood_transform
from dlmi.utils import get_device, set_seed

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAMES = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"]
NB_LAYERS_LIST = [2, 3, 5, 7, 10]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all saved model checkpoints")
    parser.add_argument("--val_path", type=str, default="data/val.h5")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--img_size", type=int, default=98)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"Device: {device}")
    print(f"Val path: {args.val_path}")
    print(f"Models dir: {args.models_dir}")

    val_preprocessing = get_ood_transform(size=args.img_size, train=False)
    val_ds = H5Dataset(args.val_path, transform=val_preprocessing, mode="train")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Val: {len(val_ds)} samples")

    results_no_tta = {}
    results_tta = {}

    for model_name in MODEL_NAMES:
        for nb_layers in NB_LAYERS_LIST:
            key = f"{model_name}_{nb_layers}_layers"
            model_path = os.path.join(args.models_dir, f"augmented_{key}.pth")

            if not os.path.exists(model_path):
                print(f"[SKIP] {model_path} not found")
                continue

            print(f"\n{'=' * 60}")
            print(f"Evaluating: {key}")
            print(f"{'=' * 60}")

            model = get_finetunable_dinov2(
                model_name, num_blocks_to_unfreeze=nb_layers, device=device
            )
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=device)
            )
            model.eval()

            acc_no_tta = evaluate_no_tta(model, val_loader, device)
            results_no_tta[key] = acc_no_tta
            print(f"  No TTA accuracy: {acc_no_tta:.4f}")

            acc_tta = evaluate_with_tta(model, args.val_path, val_preprocessing, device)
            results_tta[key] = acc_tta
            print(f"  TTA accuracy:    {acc_tta:.4f}")

    print(f"\n\nEvaluated {len(results_no_tta)} models.")
    os.makedirs(args.results_dir, exist_ok=True)

    no_tta_path = os.path.join(args.results_dir, "val_results_no_tta.json")
    tta_path = os.path.join(args.results_dir, "val_results_tta.json")

    with open(no_tta_path, "w") as f:
        json.dump(results_no_tta, f, indent=2)

    with open(tta_path, "w") as f:
        json.dump(results_tta, f, indent=2)

    print(f"Saved: {no_tta_path}")
    print(f"Saved: {tta_path}")


if __name__ == "__main__":
    main()
