"""
Precompute las features de UNI2-h para todas las images y las save en disco.

Ejecutar UNA SOLA VEZ antes de trainsr:
    python scripts/precompute_features.py

Las features se saven en:
    ./uni2_features/<image_name>.pt
    Cada archivo contiene un dict con 4 tensors: f0, f1, f2, f3
    Shape de cada tensor: [4, 1536, 36, 36]  (4 escalas)

Tiempo estimado: ~10-20 min para todo SICAPv2 en una GPU moderna.
Espacio en disco: ~8-10 GB.
"""

import warnings

import numpy as np
import cv2
cv2.setNumThreads(0)
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

import sicap_imports  # noqa: F401

from paths import IMAGES_DIR, PARTITION_DIR, uni2_features_default_cache

# ─── Paths (alineados con training_uni2_fast / artifacts) ─────────────────────
CACHE_DIR = uni2_features_default_cache()

IMG_SIZE = 504  # multiple of patch_size=14

_UNI2_TIMM_KWARGS = {
    "img_size"        : 224,
    "patch_size"      : 14,
    "depth"           : 24,
    "num_heads"       : 24,
    "init_values"     : 1e-5,
    "embed_dim"       : 1536,
    "mlp_ratio"       : 2.66667 * 2,
    "num_classes"     : 0,
    "no_embed_class"  : True,
    "mlp_layer"       : timm.layers.SwiGLUPacked,
    "act_layer"       : torch.nn.SiLU,
    "reg_tokens"      : 8,
    "dynamic_img_size": True,
}
_FEATURE_BLOCKS = {5, 11, 17, 23}
_NUM_REG        = 8
_EMBED_DIM      = 1536

# ─── Transform (sin augmentations, only normalize) ───────────────────────────
_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def load_image(path):
    buf   = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Not found: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _transform(image=image)["image"]

@torch.no_grad()
def extract_features(vit, x):
    """Devuelve lista de 4 feature maps [1, 1536, H_p, W_p]."""
    B = x.shape[0]
    x_tok = vit.patch_embed(x)
    x_tok = vit._pos_embed(x_tok)
    reg   = vit.reg_token.expand(B, -1, -1)
    x_tok = torch.cat([reg, x_tok], dim=1)
    x_tok = vit.patch_drop(x_tok)
    x_tok = vit.norm_pre(x_tok)

    num_spatial = x_tok.shape[1] - _NUM_REG
    H_p = W_p = int(num_spatial ** 0.5)

    features = []
    for i, blk in enumerate(vit.blocks):
        x_tok = blk(x_tok)
        if i in _FEATURE_BLOCKS:
            s = x_tok[:, _NUM_REG:_NUM_REG + H_p * W_p, :]
            s = s.permute(0, 2, 1).reshape(B, _EMBED_DIM, H_p, W_p)
            features.append(s.half().cpu())  # float16 para ahorrar espacio

    return features  # [f0, f1, f2, f3]

def collect_all_image_names():
    """Recoge todos los nombres de imagen de los 4 folds (train + val)."""
    names = set()
    for fold in ["Val1", "Val2", "Val3", "Val4"]:
        fold_dir = PARTITION_DIR / "Validation" / fold
        for split in ["Train.xlsx", "Test.xlsx"]:
            df = pd.read_excel(fold_dir / split)
            names.update(df["image_name"].tolist())
    return sorted(names)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CACHE_DIR.mkdir(exist_ok=True)

    print("Cargando UNI2-h...")
    vit = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **_UNI2_TIMM_KWARGS)
    vit = vit.to(device).eval()
    for p in vit.parameters():
        p.requires_grad = False
    print("✅ Modelo cargado.")

    all_names = collect_all_image_names()
    print(f"Total images a procesar: {len(all_names)}")

    # Skip those already cached
    pending = [n for n in all_names if not (CACHE_DIR / (n + ".pt")).exists()]
    print(f"Pending (not cached yet): {len(pending)}")

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for name in tqdm(pending, desc="Extrayendo features"):
        out_path = CACHE_DIR / (name + ".pt")
        try:
            img = load_image(IMAGES_DIR / name).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    feats = extract_features(vit, img)
            # Guardar como dict de tensors float16 en CPU
            torch.save({"f0": feats[0], "f1": feats[1], "f2": feats[2], "f3": feats[3]}, out_path)
        except Exception as e:
            print(f"\n⚠️  Error en {name}: {e}")

    print(f"\n✅ Features savedas en {CACHE_DIR}")
    # Estimate total size
    total_bytes = sum(f.stat().st_size for f in CACHE_DIR.glob("*.pt"))
    print(f"   Total size on disk: {total_bytes / 1e9:.2f} GB")

if __name__ == "__main__":
    main()