import torch
import torchmetrics
import torchvision.transforms.functional as F
import h5py
from tqdm import tqdm
import numpy as np


def tta_predict(model, img_tensor, device, n_augments=8):
    """Test-time augmentation: average predictions over geometric augmentations."""
    augmented = [
        img_tensor,
        F.hflip(img_tensor),
        F.vflip(img_tensor),
        F.rotate(img_tensor, 90),
        F.rotate(img_tensor, 180),
        F.rotate(img_tensor, 270),
        F.hflip(F.rotate(img_tensor, 90)),
        F.vflip(F.rotate(img_tensor, 90)),
    ][:n_augments]
    batch = torch.stack(augmented).to(device)
    with torch.no_grad():
        preds = model(batch)
    return preds.mean().item()


def evaluate_no_tta(model, val_loader, device):
    """Evaluate model on val set without TTA."""
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            val_preds.append(preds.cpu())
            val_labels.append(labels)
    val_preds = torch.cat(val_preds).squeeze()
    val_labels = torch.cat(val_labels).int()
    acc = torchmetrics.functional.accuracy(
        val_preds.round().int(), val_labels, task="binary"
    )
    return acc.item()


def evaluate_with_tta(model, val_path, val_preprocessing, device):
    """Evaluate model on val set with TTA."""
    model.eval()
    tta_preds, tta_labels = [], []
    with h5py.File(val_path, "r") as hdf:
        for img_id in tqdm(hdf.keys(), desc="  TTA", leave=False):
            img = val_preprocessing(torch.tensor(np.array(hdf[img_id]["img"])).float())
            tta_preds.append(tta_predict(model, img, device))
            tta_labels.append(int(np.array(hdf[img_id]["label"])))
    acc = torchmetrics.functional.accuracy(
        torch.tensor(tta_preds).round().int(),
        torch.tensor(tta_labels).int(),
        task="binary",
    )
    return acc.item()
