# Generate competition submission CSV files for a trained DINOv2 model.
# Runs inference both without TTA and with TTA on the test set and saves
# two CSV files to results/.

import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torch.utils.data import DataLoader

from dlmi.dataset import H5Dataset
from dlmi.model import get_finetunable_dinov2
from dlmi.test import tta_predict
from dlmi.transforms import get_ood_transform
from dlmi.utils import get_device, save_submission, set_seed

from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

IMG_SIZE = 98
BATCH_SIZE = 16
SEED = 0

set_seed(SEED)
device = get_device()
print(f"Device: {device}")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = str(REPO_ROOT / "data" / "test.h5")
val_preprocessing = get_ood_transform(size=IMG_SIZE, train=False)

MODEL_NAME = "dinov2_vitl14"
NB_LAYERS_TO_FINE_TUNE = 5
MODELS_DIR = REPO_ROOT / "models"
MODEL_SAVE_PATH = str(
    MODELS_DIR / f"augmented_{MODEL_NAME}_{NB_LAYERS_TO_FINE_TUNE}_layers.pth"
)
print(f"Loading model from {MODEL_SAVE_PATH}")

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = str(
    RESULTS_DIR / f"submission_{MODEL_NAME}_{NB_LAYERS_TO_FINE_TUNE}_layers.csv"
)

model = get_finetunable_dinov2(
    MODEL_NAME,
    num_blocks_to_unfreeze=NB_LAYERS_TO_FINE_TUNE,
    device=device,
    use_mixstyle=True,
)
model.load_state_dict(
    torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
)
model.eval()

test_ds = H5Dataset(TEST_PATH, transform=val_preprocessing, mode="test")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

test_preds_no_tta = []
with torch.no_grad():
    for imgs in tqdm(test_loader, desc="Test (no TTA)"):
        imgs = imgs.to(device)
        preds = model(imgs)
        test_preds_no_tta.append(preds.cpu())

test_preds_no_tta = torch.cat(test_preds_no_tta).squeeze().tolist()
test_ids_no_tta = [int(i) for i in test_ds.image_ids]

submission_no_tta = save_submission(
    test_ids_no_tta, test_preds_no_tta, SUBMISSION_PATH.replace(".csv", "_no_tta.csv")
)
print(f"Submission (no TTA) saved ({len(submission_no_tta)} rows)")

model = get_finetunable_dinov2(
    MODEL_NAME,
    num_blocks_to_unfreeze=NB_LAYERS_TO_FINE_TUNE,
    device=device,
    use_mixstyle=True,
)
model.load_state_dict(
    torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
)
model.eval()

test_ids, test_preds = [], []

with h5py.File(TEST_PATH, "r") as hdf:
    for test_id in tqdm(hdf.keys(), desc="Test TTA"):
        img = val_preprocessing(torch.tensor(np.array(hdf[test_id]["img"])).float())
        pred = tta_predict(model, img, device)
        test_ids.append(int(test_id))
        test_preds.append(pred)

submission = save_submission(
    test_ids, test_preds, SUBMISSION_PATH.replace(".csv", "_tta.csv")
)
print(f"Submission saved to {SUBMISSION_PATH} ({len(submission)} rows)")
