import torch
import cv2
import numpy as np
from training_conchv2 import CONCHSegModel, _MASK_LUT, get_val_transforms
from pathlib import Path

BASE_DIR = Path(".")
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
CHECKPOINT_PATH = BASE_DIR / "checkpoints_conch_masklut" / "best_Val1_0.6784.pth"

model = CONCHSegModel(num_classes=4, unfreeze_last=0)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
model.eval()

image_name = "18B0003808K_Block_Region_7_22_4_xini_15996_yini_39887.jpg"
image = cv2.cvtColor(cv2.imdecode(np.fromfile(str(IMAGES_DIR / image_name), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
mask_raw = cv2.imdecode(np.fromfile(str(MASKS_DIR / image_name), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
mask = _MASK_LUT[mask_raw]

transformed = get_val_transforms()(image=image, mask=mask)
with torch.no_grad():
    pred = model(transformed["image"].unsqueeze(0)).argmax(dim=1).squeeze(0).numpy()
mask_t = transformed["mask"].numpy()

print(f"Mask counts: 0:{np.sum(mask_t==0)}  1:{np.sum(mask_t==1)}  2:{np.sum(mask_t==2)}  3:{np.sum(mask_t==3)}")
print(f"Pred counts: 0:{np.sum(pred==0)}  1:{np.sum(pred==1)}  2:{np.sum(pred==2)}  3:{np.sum(pred==3)}")

def calc_f1(c):
    tp = np.sum((pred==c) & (mask_t==c))
    fp = np.sum((pred==c) & (mask_t!=c))
    fn = np.sum((pred!=c) & (mask_t==c))
    return (2.0*tp)/(2.0*tp + fp + fn) if tp+fp+fn>0 else "NaN"

print(f"F1 0: {calc_f1(0)}")
print(f"F1 1: {calc_f1(1)}")
print(f"F1 2: {calc_f1(2)}")
print(f"F1 3: {calc_f1(3)}")
