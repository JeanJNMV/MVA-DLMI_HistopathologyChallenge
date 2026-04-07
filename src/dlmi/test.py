import torch
import torchmetrics
import h5py
from tqdm import tqdm
import numpy as np

from dlmi.transforms import get_d4_transforms


def tta_predict(model, img_tensor, device, n_augments=8):
    augmented = get_d4_transforms(img_tensor)[:n_augments]
    batch = torch.stack(augmented).to(device)
    with torch.no_grad():
        preds = model(batch)
    return preds.mean().item()


def evaluate_no_tta(model, val_loader, device):
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
