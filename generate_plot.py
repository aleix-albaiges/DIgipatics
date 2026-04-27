import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from training_conchv2 import CONCHSegModel, _MASK_LUT, get_val_transforms

BASE_DIR = Path(".")
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
CHECKPOINT_PATH = BASE_DIR / "checkpoints_conch_masklut" / "best_Val1_0.6784.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

print("Loading model...")
model = CONCHSegModel(num_classes=NUM_CLASSES, unfreeze_last=0)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
transforms = get_val_transforms()

def load_image_and_mask(image_name):
    img_path = IMAGES_DIR / image_name
    buf = np.fromfile(str(img_path), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_path = MASKS_DIR / image_name
    buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
    mask_raw = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
    mask = _MASK_LUT[mask_raw]
    return image, mask

image_name = "18B0003808K_Block_Region_7_22_4_xini_15996_yini_39887.jpg"
img, mask = load_image_and_mask(image_name)
transformed = transforms(image=img, mask=mask)
img_t = transformed["image"].unsqueeze(0).to(DEVICE)
mask_t = transformed["mask"].numpy()

with torch.no_grad():
    logits = model(img_t)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

diff = (pred != mask_t).astype(np.float32)
diff = np.where(diff == 1, 1, 0)
mask_alpha = np.where(mask_t == 0, 0.0, 0.5)
pred_alpha = np.where(pred == 0, 0.0, 0.5)
diff_alpha = np.where(diff == 1, 0.5, 0.0)

img_disp = transformed["image"].permute(1, 2, 0).cpu().numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_disp = std * img_disp + mean
img_disp = np.clip(img_disp, 0, 1)

cmap = mcolors.ListedColormap(['black', 'green', 'yellow', 'red'])
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

fig, axs = plt.subplots(1, 4, figsize=(22, 6))
axs[0].imshow(img_disp)
axs[0].set_title("Original")
axs[1].imshow(img_disp)
axs[1].imshow(mask_t, cmap=cmap, norm=norm, alpha=mask_alpha)
axs[1].set_title("Ground Truth")
axs[2].imshow(img_disp)
axs[2].imshow(pred, cmap=cmap, norm=norm, alpha=pred_alpha)
axs[2].set_title("Pred")
axs[3].imshow(img_disp)
axs[3].imshow(diff, cmap="Reds", alpha=diff_alpha)
axs[3].set_title("Diff")

plt.savefig("test_plot.png")
print("Saved test_plot.png")
