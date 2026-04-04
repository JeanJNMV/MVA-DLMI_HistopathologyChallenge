import torch
import torchmetrics
import h5py
from tqdm import tqdm
import numpy as np

from dlmi.transforms import get_d4_transforms


def tta_predict(model, img_tensor, device, n_augments=8):
    """Predict with test-time augmentation.

    Averages predictions over geometric augmentations (flips and rotations).

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.
    img_tensor : torch.Tensor
        Single image tensor of shape "(C, H, W)".
    device : torch.device
        Device to run inference on.
    n_augments : int, optional
        Number of geometric augmentations to use (max 8).

    Returns
    -------
    float
        Mean predicted probability across augmentations.
    """
    augmented = get_d4_transforms(img_tensor)[:n_augments]
    batch = torch.stack(augmented).to(device)
    with torch.no_grad():
        preds = model(batch)
    return preds.mean().item()


# TTA other version with Jittering
# def tta_predict(model, img_tensor, device, n_augments=8, n_stain=3):
#     hed_jitter = HEDJitter(theta=0.05)

#     geometric = [
#         img_tensor,
#         F.hflip(img_tensor),
#         F.vflip(img_tensor),
#         F.rotate(img_tensor, 90),
#         F.rotate(img_tensor, 180),
#         F.rotate(img_tensor, 270),
#         F.hflip(F.rotate(img_tensor, 90)),
#         F.vflip(F.rotate(img_tensor, 90)),
#     ][:n_augments]

#     # Add stain-jittered copies of the original
#     stain_variants = [hed_jitter(img_tensor) for _ in range(n_stain)]

#     all_variants = geometric + stain_variants
#     batch = torch.stack(all_variants).to(device)
#     with torch.no_grad():
#         preds = model(batch)
#     return preds.mean().item()


def evaluate_no_tta(model, val_loader, device):
    """Evaluate model on a validation set without test-time augmentation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.
    val_loader : torch.utils.data.DataLoader
        Dataloader yielding "(images, labels)" batches.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    float
        Binary accuracy on the validation set.
    """
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
    """Evaluate model on a validation set with test-time augmentation.

    Reads images directly from the HDF5 file and applies TTA to each sample.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.
    val_path : str
        Path to the validation HDF5 file.
    val_preprocessing : callable
        Preprocessing transform applied before augmentation.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    float
        Binary accuracy on the validation set with TTA.
    """
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
