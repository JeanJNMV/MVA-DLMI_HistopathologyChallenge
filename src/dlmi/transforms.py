import random

import torch
import torchvision.transforms as T

_HIBOU_MODELS = {"hibou-b", "hibou-l"}
_HIBOU_MEAN = [0.7068, 0.5755, 0.7220]
_HIBOU_STD = [0.1950, 0.2316, 0.1816]


def get_default_img_size(model_name="dinov2_vits14"):
    return 224 if model_name in _HIBOU_MODELS else 98


def get_baseline_transform(size=98):
    return T.Resize((size, size))


class ToUnitInterval:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0
        return img.clamp(0.0, 1.0)


class HEDJitter:
    RGB2HED = torch.tensor(
        [
            [0.6500286, 0.70436568, 0.28603244],
            [0.70427212, -0.7366021, 0.05810294],
            [0.28368167, 0.06436904, -0.95681546],
        ],
        dtype=torch.float32,
    )
    HED2RGB = torch.linalg.inv(RGB2HED)

    def __init__(self, theta=0.05):
        self.theta = theta

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.max() > 1.0:
            img = img / 255.0
        img = img.clamp(1e-6, 1.0)

        od = -torch.log(img)
        od_flat = od.view(3, -1)
        hed = self.RGB2HED @ od_flat

        alpha = torch.empty(3, 1).uniform_(1 - self.theta, 1 + self.theta)
        beta = torch.empty(3, 1).uniform_(-self.theta, self.theta)
        hed = hed * alpha + beta

        od_aug = self.HED2RGB @ hed
        img_aug = torch.exp(-od_aug).view(3, img.shape[1], img.shape[2])
        return img_aug.clamp(0.0, 1.0)


def extract_hed_stats(img_tensor: torch.Tensor) -> torch.Tensor:
    """Extract mean HED vector from a single image.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Image tensor of shape (C, H, W) with values in [0, 1].

    Returns
    -------
    torch.Tensor
        HED mean vector of shape (3, 1).
    """
    RGB2HED = HEDJitter.RGB2HED  # reuse the matrix you already defined
    img = img_tensor.clamp(1e-6, 1.0)
    od = -torch.log(img)
    od_flat = od.view(3, -1)
    hed = RGB2HED @ od_flat  # (3, H*W)
    return hed.mean(dim=1, keepdim=True)  # (3, 1) — compact stain signature


def build_stain_bank(dataset, max_images: int = 500) -> list[torch.Tensor]:
    """Collect HED stain signatures from a subset of training images.

    Call this ONCE after creating your dataset, before training starts.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Training dataset returning (image_tensor, label) pairs.
        Images should already be resized and in [0, 1].
    max_images : int, optional
        Maximum number of images to sample for the bank.

    Returns
    -------
    list[torch.Tensor]
        List of HED mean vectors, each of shape (3, 1).
    """
    bank = []
    indices = torch.randperm(len(dataset))[:max_images]
    for i in indices.tolist():
        img, _ = dataset[i]
        if img.max() > 1.0:
            img = img / 255.0
        bank.append(extract_hed_stats(img))
    return bank


class StainMix:
    """Mix the stain signature of an image with one drawn from a stain bank.

    Simulates seeing the same tissue under a different center's staining
    protocol by interpolating HED statistics between the source image and
    a randomly sampled reference from the bank.

    Parameters
    ----------
    stain_bank : list[torch.Tensor]
        List of HED mean vectors of shape (3, 1), built by build_stain_bank().
    alpha : float, optional
        Maximum interpolation weight toward the reference stain.
        0.0 = no change, 1.0 = full replacement.
    """

    HED2RGB = HEDJitter.HED2RGB  # reuse matrices already defined
    RGB2HED = HEDJitter.RGB2HED

    def __init__(self, stain_bank: list, alpha: float = 0.3):
        self.stain_bank = stain_bank
        self.alpha = alpha

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply stain mixing to an image.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor of shape (C, H, W) with values in [0, 1].

        Returns
        -------
        torch.Tensor
            Stain-mixed image tensor of shape (C, H, W) clamped to [0, 1].
        """
        if img.max() > 1.0:
            img = img / 255.0
        img = img.clamp(1e-6, 1.0)

        od = -torch.log(img)
        od_flat = od.view(3, -1)
        hed = self.RGB2HED @ od_flat  # (3, H*W)

        # Source stain signature
        src_mean = hed.mean(dim=1, keepdim=True)  # (3, 1)

        # Random reference stain from bank
        ref_mean = random.choice(self.stain_bank).to(img.device)  # (3, 1)

        # Interpolation weight — sample fresh each call
        lam = random.uniform(0, self.alpha)

        # Shift HED values toward the reference stain center
        hed_mixed = hed - src_mean + (1 - lam) * src_mean + lam * ref_mean

        od_aug = self.HED2RGB @ hed_mixed
        img_aug = torch.exp(-od_aug).view(3, img.shape[1], img.shape[2])
        return img_aug.clamp(0.0, 1.0)


def get_d4_transforms(img: torch.Tensor):
    rotations = [torch.rot90(img, k, dims=(-2, -1)) for k in range(4)]
    return rotations + [torch.flip(rot, dims=(-1,)) for rot in rotations]


class RandomD4:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return get_d4_transforms(img)[random.randrange(8)]


def get_ood_transform(size=None, train=True, model_name="dinov2_vits14", stain_bank=None):
    """Return an OOD-robust transform pipeline.

    Parameters
    ----------
    size : int or None, optional
        Target height and width in pixels.  Defaults to the model's native size.
    train : bool, optional
        If True, include augmentation transforms.
    model_name : str, optional
        Model name used to select Hibou-specific normalisation.
    stain_bank : list[torch.Tensor] or None, optional
        Stain bank built by build_stain_bank(). When provided during training,
        a StainMix step is added to the pipeline.

    Returns
    -------
    torchvision.transforms.Compose
        Composed transform pipeline.
    """
    size = get_default_img_size(model_name) if size is None else size

    if model_name in _HIBOU_MODELS:
        transforms = [T.Resize((size, size)), ToUnitInterval()]
        if train:
            transforms.append(HEDJitter(theta=0.05))
            if stain_bank is not None:
                transforms.append(StainMix(stain_bank, alpha=0.3))
            transforms.append(RandomD4())
        transforms.append(T.Normalize(mean=_HIBOU_MEAN, std=_HIBOU_STD))
        return T.Compose(transforms)

    if train:
        augs = [
            T.Resize((size, size)),
            HEDJitter(theta=0.05),
        ]
        if stain_bank is not None:
            augs.append(StainMix(stain_bank, alpha=0.3))
        augs.append(RandomD4())
        return T.Compose(augs)
    return T.Resize((size, size))
