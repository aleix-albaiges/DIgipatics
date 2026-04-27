import torch
import cv2
import numpy as np
from pathlib import Path
from training_conchv2 import CONCHSegModel, _MASK_LUT, get_val_transforms

BASE_DIR = Path(".")
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
CHECKPOINT_PATH = BASE_DIR / "checkpoints_conch_masklut" / "best_Val1_0.6784.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

model = CONCHSegModel(num_classes=NUM_CLASSES, unfreeze_last=0)
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

transforms = get_val_transforms()

image_name = "18B0003808K_Block_Region_7_22_4_xini_15996_yini_39887.jpg"
img_path = IMAGES_DIR / image_name
mask_path = MASKS_DIR / image_name

buf = np.fromfile(str(img_path), dtype=np.uint8)
image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
mask_raw = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
mask = _MASK_LUT[mask_raw]

transformed = transforms(image=image, mask=mask)
img_t = transformed["image"].unsqueeze(0).to(DEVICE)
mask_t = transformed["mask"].numpy()

with torch.no_grad():
    logits = model(img_t)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

diff = (pred != mask_t).astype(np.float32)

print("Unique values in mask_raw:", np.unique(mask_raw))
print("Unique values in mask:", np.unique(mask))
print("Unique values in mask_t:", np.unique(mask_t))
print("Unique values in pred:", np.unique(pred))
print("Number of pixels where pred == mask_t:", np.sum(pred == mask_t))
print("Number of pixels where pred != mask_t:", np.sum(pred != mask_t))
