"""Environment smoke test — run this before any full training job on Ruche.

Checks:
  1. Python version and key imports
  2. GPU availability and VRAM
  3. HDF5 data files readable
  4. Model loads and forward pass works
  5. DataLoader produces correct batches

Usage
-----
    python scripts/smoke_test.py --train_path data/train.h5 --val_path data/val.h5
    python scripts/smoke_test.py --model_name hibou-l  # also tests HuggingFace download
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="data/train.h5")
parser.add_argument("--val_path", type=str, default="data/val.h5")
parser.add_argument("--model_name", type=str, default="dinov2_vits14")
parser.add_argument("--num_unfreeze", type=int, default=2)
args = parser.parse_args()

PASS = "[OK]"
FAIL = "[FAIL]"


def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  {FAIL}  {label} — {e}")
        return False


print("\n=== Smoke test ===\n")

# 1. Python
print("1. Python / imports")
check("Python version", lambda: sys.version.split()[0])
check("torch", lambda: __import__("torch").__version__)
check("h5py", lambda: __import__("h5py").__version__)
check("timm", lambda: __import__("timm").__version__)
check("transformers", lambda: __import__("transformers").__version__)
check("torchmetrics", lambda: __import__("torchmetrics").__version__)
check("dlmi.dataset", lambda: __import__("dlmi.dataset"))
check("dlmi.model", lambda: __import__("dlmi.model"))
check("dlmi.train", lambda: __import__("dlmi.train"))

# 2. GPU
print("\n2. GPU")
import torch
check("CUDA available", lambda: str(torch.cuda.is_available()))
if torch.cuda.is_available():
    check("GPU name", lambda: torch.cuda.get_device_name(0))
    check(
        "VRAM",
        lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
    )

# 3. Data
print("\n3. Data files")
from dlmi.dataset import H5Dataset
from dlmi.transforms import get_default_img_size, get_ood_transform

img_size = get_default_img_size(args.model_name)
print(f"Using image size: {img_size}")

transform = get_ood_transform(size=img_size, train=False, model_name=args.model_name)

def _check_h5(path):
    ds = H5Dataset(path, transform=transform, mode="train")
    img, label = ds[0]
    return f"{len(ds)} samples, img shape {tuple(img.shape)}, label {label}"

check(f"train.h5 ({args.train_path})", lambda: _check_h5(args.train_path))
check(f"val.h5   ({args.val_path})", lambda: _check_h5(args.val_path))

# 4. Model
print(f"\n4. Model ({args.model_name}, unfreeze={args.num_unfreeze})")
from dlmi.model import get_finetunable_dinov2
from dlmi.utils import get_device

device = get_device()

def _check_model():
    t0 = time.time()
    model = get_finetunable_dinov2(
        args.model_name, num_blocks_to_unfreeze=args.num_unfreeze, device=device
    )
    elapsed = time.time() - t0
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return f"loaded in {elapsed:.1f}s | trainable {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M params"

check("model load", _check_model)

# 5. Forward pass + DataLoader
print("\n5. Forward pass")
from torch.utils.data import DataLoader

def _check_forward():
    model = get_finetunable_dinov2(
        args.model_name, num_blocks_to_unfreeze=args.num_unfreeze, device=device
    )
    ds = H5Dataset(args.train_path, transform=transform, mode="train")
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    x, y = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        logits = model.forward_logits(x)
    return f"input {tuple(x.shape)} → logits {tuple(logits.shape)}"

check("forward pass", _check_forward)

print("\n=== Done ===\n")
