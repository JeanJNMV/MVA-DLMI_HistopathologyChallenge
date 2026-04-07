import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class H5Dataset(Dataset):
    def __init__(self, dataset_path, transform=None, mode="train", centers=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.mode = mode
        self._hdf = None

        with h5py.File(self.dataset_path, "r") as hdf:
            if centers is not None:
                self.image_ids = [
                    img_id
                    for img_id in hdf.keys()
                    if int(np.array(hdf[img_id]["metadata"])[0]) in centers
                ]
            else:
                self.image_ids = list(hdf.keys())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_hdf"] = None
        return state

    def _get_hdf(self):
        if self._hdf is None:
            self._hdf = h5py.File(self.dataset_path, "r")
        return self._hdf

    def close(self):
        if self._hdf is not None:
            hdf = self._hdf
            self._hdf = None
            try:
                hdf.close()
            except (AttributeError, TypeError, ValueError):
                pass

    def __del__(self):
        self.close()

    def __len__(self):
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
            Label array when "mode='train'", otherwise only the image is returned.
        """
        img_id = self.image_ids[idx]
        grp = self._get_hdf()[img_id]
        img = torch.from_numpy(np.asarray(grp["img"])).float()
        label = int(np.asarray(grp["label"])) if self.mode == "train" else -1
        if self.transform:
            img = self.transform(img)
        if self.mode == "train":
            return img, label
        else:
            return img


class PrecomputedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()


def precompute_features(dataloader, model, device):
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
    ids, labels, centers = [], [], []
    with h5py.File(dataset_path, "r") as hdf:
        for img_id in hdf.keys():
            ids.append(int(img_id))
            grp = hdf.get(img_id)
            labels.append(int(np.array(grp.get("label"))))
            centers.append(int(np.array(grp.get("metadata"))[0]))
    return ids, labels, centers
