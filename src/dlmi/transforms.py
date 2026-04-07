import torch
import torchvision.transforms as T
import random


def get_baseline_transform(size=98):
    """Return a baseline preprocessing transform that resizes to a fixed square.

    Parameters
    ----------
    size : int, optional
        Target height and width in pixels.

    Returns
    -------
    torchvision.transforms.Resize
        Resize transform.
    """
    return T.Resize((size, size))


class HEDJitter:
    """Randomly perturb H&E stain intensities in HED color space.

    Simulates inter-center staining variability by applying random affine
    perturbations to each channel of the HED decomposition.

    Parameters
    ----------
    theta : float, optional
        Magnitude of stain perturbation. 0.05 is subtle, 0.15 is strong.

    References
    ----------
    Tellez et al., "Quantifying the effects of data augmentation and stain
    color normalization in convolutional neural networks for computational
    pathology", Medical Image Analysis, 2019.
    """

    # RGB -> HED color deconvolution matrix (standard H&E)
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
        """Initialize HEDJitter.

        Parameters
        ----------
        theta : float, optional
            Magnitude of stain perturbation.
        """
        self.theta = theta

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random HED stain jittering to an image.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor of shape "(C, H, W)" with values in "[0, 255]"
            or "[0, 1]".

        Returns
        -------
        torch.Tensor
            Augmented image tensor of shape "(C, H, W)" clamped to
            "[0, 1]".
        """
        # img: (C, H, W) float32, values in [0, 255] or [0, 1]
        # Normalize to [0,1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        img = img.clamp(1e-6, 1.0)

        # Convert RGB -> OD (optical density)
        od = -torch.log(img)  # (3, H, W)

        # Project to HED space: (3, H*W)
        od_flat = od.view(3, -1)
        hed = self.RGB2HED @ od_flat  # (3, H*W)

        # Perturb each stain channel independently
        alpha = torch.empty(3, 1).uniform_(1 - self.theta, 1 + self.theta)
        beta = torch.empty(3, 1).uniform_(-self.theta, self.theta)
        hed = hed * alpha + beta

        # Back to RGB
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


def get_ood_transform(size=98, train=True, stain_bank=None):
    """Return an OOD-robust transform pipeline.

    Parameters
    ----------
    size : int, optional
        Target height and width in pixels.
    train : bool, optional
        If True, include augmentation transforms.
    stain_bank : list[torch.Tensor] or None, optional
        Stain bank built by build_stain_bank(). Required for StainMix
        during training. If None, StainMix is skipped.

    Returns
    -------
    torchvision.transforms.Compose
        Composed transform pipeline.
    """
    if train:
        augs = [
            T.Resize((size, size)),
            HEDJitter(theta=0.05),
        ]
        if stain_bank is not None:
            augs.append(StainMix(stain_bank, alpha=0.3))
        augs += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
        ]
        return T.Compose(augs)
    else:
        return T.Resize((size, size))
