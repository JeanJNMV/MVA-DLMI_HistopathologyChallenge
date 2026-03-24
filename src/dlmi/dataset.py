import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.notebook import tqdm


class H5Dataset(Dataset):
    """Dataset for reading histopathology patches from HDF5 files.

    Each sample is lazily loaded from disk on access.

    Parameters
    ----------
    dataset_path : str
        Path to the HDF5 file containing images and labels.
    transform : callable or None, optional
        Transform to apply to each image tensor.
    mode : str, optional
        If "'train'", labels are loaded; otherwise labels are "None".
    """

    def __init__(self, dataset_path, transform=None, mode="train"):
        self.dataset_path = dataset_path
        self.transform = transform
        self.mode = mode

        with h5py.File(self.dataset_path, "r") as hdf:
            self.image_ids = list(hdf.keys())

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of images.
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """Load and return a single sample.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        img : torch.Tensor
            Image tensor of shape "(C, H, W)".
        label : numpy.ndarray or None
            Label array when "mode='train'", otherwise "None".
        """
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, "r") as hdf:
            img = torch.tensor(hdf.get(img_id).get("img")).float()
            label = (
                np.array(hdf.get(img_id).get("label")) if self.mode == "train" else None
            )
        if self.transform:
            img = self.transform(img)
        return img, label


class PrecomputedDataset(Dataset):
    """Dataset wrapping precomputed feature embeddings and labels.

    Parameters
    ----------
    features : torch.Tensor
        Feature matrix of shape "(N, D)".
    labels : torch.Tensor
        Label vector of shape "(N,)".
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.unsqueeze(-1)

    def __len__(self):
        """Return the number of samples.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a single feature-label pair.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        features : torch.Tensor
            Feature vector of shape "(D,)".
        label : torch.Tensor
            Label scalar as a float tensor.
        """
        return self.features[idx], self.labels[idx].float()


def precompute_features(dataloader, model, device):
    """Extract features from a pretrained model for all samples in a dataloader.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader yielding "(images, labels)" batches.
    model : torch.nn.Module
        Pretrained feature extractor.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    features : torch.Tensor
        Stacked feature matrix of shape "(N, D)".
    labels : torch.Tensor
        Concatenated label vector of shape "(N,)".
    """
    xs, ys = [], []
    model.eval()
    for x, y in tqdm(dataloader, leave=False, desc="Precomputing features"):
        with torch.no_grad():
            xs.append(model(x.to(device)).detach().cpu().numpy())
        ys.append(y.numpy())
    xs = np.vstack(xs)
    ys = np.hstack(ys)
    return torch.tensor(xs), torch.tensor(ys)


def load_h5_metadata(dataset_path):
    """Load image IDs, labels, and center info from an HDF5 file.

    Parameters
    ----------
    dataset_path : str
        Path to the HDF5 file.

    Returns
    -------
    ids : list of int
        Image identifiers.
    labels : list of int
        Binary labels for each image.
    centers : list of int
        Center identifier for each image.
    """
    ids, labels, centers = [], [], []
    with h5py.File(dataset_path, "r") as hdf:
        for img_id in hdf.keys():
            ids.append(int(img_id))
            grp = hdf.get(img_id)
            labels.append(int(np.array(grp.get("label"))))
            centers.append(int(np.array(grp.get("metadata"))[0]))
    return ids, labels, centers
