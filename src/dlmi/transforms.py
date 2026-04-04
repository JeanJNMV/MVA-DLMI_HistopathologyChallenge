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


def get_d4_transforms(img: torch.Tensor):
    """Return the 8 exact dihedral symmetries of an image tensor."""
    rotations = [torch.rot90(img, k, dims=(-2, -1)) for k in range(4)]
    return rotations + [torch.flip(rot, dims=(-1,)) for rot in rotations]


class RandomD4:
    """Sample one exact rotation/flip symmetry from the D4 group."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return get_d4_transforms(img)[random.randrange(8)]


def get_ood_transform(size=98, train=True):
    """Return an OOD-robust transform pipeline.

    For training, applies resize, HED stain jittering, and one exact D4
    symmetry (rotations by 90 degrees and flips). For validation and test,
    applies resize only.

    Parameters
    ----------
    size : int, optional
        Target height and width in pixels.
    train : bool, optional
        If "True", include augmentation transforms.

    Returns
    -------
    torchvision.transforms.Compose or torchvision.transforms.Resize
        Composed transform pipeline.
    """
    if train:
        return T.Compose(
            [
                T.Resize((size, size)),
                HEDJitter(theta=0.05),  # stain aug
                RandomD4(),
            ]
        )
    else:
        return T.Resize((size, size))  # no aug at inference
