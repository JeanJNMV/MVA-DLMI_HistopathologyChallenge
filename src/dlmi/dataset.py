import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm


class H5Dataset(Dataset):
    """Dataset for reading histopathology patches from HDF5 files."""

    def __init__(self, dataset_path, transform=None, mode="train"):
        self.dataset_path = dataset_path
        self.transform = transform
        self.mode = mode

        with h5py.File(self.dataset_path, "r") as hdf:
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
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
    """Dataset wrapping precomputed feature embeddings and labels."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()


def precompute_features(dataloader, model, device):
    """Extract features from a pretrained model for all samples in a dataloader."""
    xs, ys = [], []
    model.eval()
    for x, y in tqdm(dataloader, leave=False, desc="Precomputing features"):
        with torch.no_grad():
            xs.append(model(x.to(device)).detach().cpu().numpy())
        ys.append(y.numpy())
    xs = np.vstack(xs)
    ys = np.hstack(ys)
    return torch.tensor(xs), torch.tensor(ys)


def get_dataloader(dataset, batch_size=16, shuffle=True):
    """Create a DataLoader from a dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_h5_metadata(dataset_path):
    """Load image IDs, labels, and center info from an HDF5 file."""
    ids, labels, centers = [], [], []
    with h5py.File(dataset_path, "r") as hdf:
        for img_id in hdf.keys():
            ids.append(int(img_id))
            grp = hdf.get(img_id)
            labels.append(int(np.array(grp.get("label"))))
            centers.append(int(np.array(grp.get("metadata"))[0]))
    return ids, labels, centers
