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


def get_d4_transforms(img: torch.Tensor):
    rotations = [torch.rot90(img, k, dims=(-2, -1)) for k in range(4)]
    return rotations + [torch.flip(rot, dims=(-1,)) for rot in rotations]


class RandomD4:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return get_d4_transforms(img)[random.randrange(8)]


def get_ood_transform(size=None, train=True, model_name="dinov2_vits14"):
    size = get_default_img_size(model_name) if size is None else size

    if model_name in _HIBOU_MODELS:
        transforms = [T.Resize((size, size)), ToUnitInterval()]
        if train:
            transforms.extend([HEDJitter(theta=0.05), RandomD4()])
        transforms.append(T.Normalize(mean=_HIBOU_MEAN, std=_HIBOU_STD))
        return T.Compose(transforms)

    if train:
        return T.Compose(
            [
                T.Resize((size, size)),
                HEDJitter(theta=0.05),
                RandomD4(),
            ]
        )
    return T.Resize((size, size))
