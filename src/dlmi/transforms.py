import torchvision.transforms as T


def get_baseline_transform(size=98):
    """Baseline preprocessing: resize to a fixed square size."""
    return T.Resize((size, size))
