import json

import sicap_imports  # noqa: F401
from sicap_imports import REPO_ROOT

nb_path = str(REPO_ROOT / "notebooks" / "analisis_fscore_checkpoints.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_code = """import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

print("Calculando cantidad de pixeles reales (Ground Truth) por imagen...")
pixel_counts = []

for name in tqdm(df_results['image_name']):
    mask_path = MASKS_DIR / name
    if not mask_path.exists():
        pixel_counts.append({'image_name': name, 'pixels_NC': 0, 'pixels_GG3': 0, 'pixels_GG4': 0, 'pixels_GG5': 0})
        continue
        
    buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
    mask_raw = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        pixel_counts.append({'image_name': name, 'pixels_NC': 0, 'pixels_GG3': 0, 'pixels_GG4': 0, 'pixels_GG5': 0})
        continue
        
    # Usar _MASK_LUT original para tener los valores correctos (0, 1, 2, 3)
    mask = _MASK_LUT[mask_raw]
    
    # np.bincount es la fastest way de contar (0 a 3)
    counts = np.bincount(mask.flatten(), minlength=4)
    
    pixel_counts.append({
        'image_name': name, 
        'pixels_NC': counts[0], 
        'pixels_GG3': counts[1], 
        'pixels_GG4': counts[2], 
        'pixels_GG5': counts[3]
    })

# Convertir a DataFrame y merge por 'image_name' para evitar duplicados si se corre varias veces
df_pixels = pd.DataFrame(pixel_counts)

if 'pixels_NC' in df_results.columns:
    df_results = df_results.drop(columns=['pixels_NC', 'pixels_GG3', 'pixels_GG4', 'pixels_GG5', 'pixels_cancer', 'cancer_percentage'], errors='ignore')

df_results = pd.merge(df_results, df_pixels, on='image_name', how='left')

# Calcular tumor total (GG3 + GG4 + GG5)
df_results['pixels_cancer'] = df_results['pixels_GG3'] + df_results['pixels_GG4'] + df_results['pixels_GG5']
df_results['cancer_percentage'] = (df_results['pixels_cancer'] / (512 * 512)) * 100

print("¡Columnas de pixeles añadidas al df_results original!")
print("-" * 50)

# Apply advanced filter para encontrar images GIGANTES y con mala métrica
df_interesante = df_results[(df_results['pixels_cancer'] > 50000) & (df_results['macro_f1'] < 0.5)]
df_interesante = df_interesante.sort_values(by='pixels_cancer', ascending=False)

print(f"Se han encontrado {len(df_interesante)} images con más de 50.000 pixeles de tumor y un Macro F1 < 0.5")
display(df_interesante[['image_name', 'macro_f1', 'pixels_cancer', 'cancer_percentage', 'f1_GG3', 'f1_GG4', 'f1_GG5']].head(10))

# Uncomment la siguiente línea para visualizar directamente la que tenga MOST tumor pero peor F1:
# plot_image_analysis(df_interesante.iloc[0]['image_name'], df_interesante.iloc[0]['macro_f1'])"""

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" if i < len(new_code.split('\n')) - 1 else line for i, line in enumerate(new_code.split('\n'))]
}

nb["cells"].append(new_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Appended cell to notebook using json logic.")
