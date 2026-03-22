import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F


def get_baseline_transform(size=98):
    """Baseline preprocessing: resize to a fixed square size."""
    return T.Resize((size, size))


class HEDJitter:
    """
    Randomly perturbs H&E stain intensities in HED color space.
    This directly simulates inter-center staining variability.

    Reference: Tellez et al., "Quantifying the effects of data augmentation
    and stain color normalization in convolutional neural networks for
    computational pathology", Medical Image Analysis 2019.
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
        """theta: magnitude of stain perturbation (0.05 is subtle, 0.15 is strong)"""
        self.theta = theta

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
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


def get_ood_transform(size=98, train=True):
    """OOD-robust transform with stain augmentation for train, clean for val/test."""
    if train:
        return T.Compose(
            [
                T.Resize((size, size)),
                HEDJitter(theta=0.05),  # stain aug
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(90),
            ]
        )
    else:
        return T.Resize((size, size))  # no aug at inference
