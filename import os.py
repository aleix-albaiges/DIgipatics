import os
import pandas as pd

BASE = r"C:\Users\Aleix\OneDrive - Universitat Politècnica de Catalunya\Escritorio\UNI\TFG\Recerca primers datasets\SicapV2\SICAPv2"

# carregar el mapeig slide→patient
wsi = pd.read_excel(os.path.join(BASE, "wsi_labels.xlsx"))

# construir el df global com abans
dfs = []
for fold in ['Val1', 'Val2', 'Val3', 'Val4']:
    for split in ['Train', 'Test']:
        df_tmp = pd.read_excel(os.path.join(BASE, "partition", "Validation", fold, f"{split}.xlsx"))
        df_tmp['fold'] = fold
        df_tmp['split'] = split
        dfs.append(df_tmp)
for split in ['Train', 'Test']:
    df_tmp = pd.read_excel(os.path.join(BASE, "partition", "Test", f"{split}.xlsx"))
    df_tmp['fold'] = 'Test_oficial'
    df_tmp['split'] = split
    dfs.append(df_tmp)

df_all = pd.concat(dfs, ignore_index=True)

# extreure slide_id i unir amb wsi_labels per obtenir el veritable patient_id
df_all['slide_id'] = df_all['image_name'].str.split('_Block').str[0]
df_all = df_all.merge(wsi[['slide_id', 'patient_id']], on='slide_id', how='left')

# comprovar que no hi hagi valors perduts
if df_all['patient_id'].isna().any():
    print("!! imatges sense pacient en wsi_labels:", df_all.loc[df_all['patient_id'].isna(), 'slide_id'].unique())

# etiqueta única
df_all['label'] = df_all[['NC', 'G3', 'G4', 'G5']].idxmax(axis=1)

# nombre de pacients reals
unique_patients = df_all['patient_id'].nunique()
print("Pacients únics en tot el dataset:", unique_patients)
assert unique_patients == 95, "no coincideixen els 95 pacients esperats"

# resum per fold/split
for fold in ['Val1', 'Val2', 'Val3', 'Val4']:
    df_fold = df_all[df_all['fold'] == fold]
    train = df_fold[df_fold['split'] == 'Train']
    test  = df_fold[df_fold['split'] == 'Test']

    pts_train = set(train['patient_id'])
    pts_test  = set(test['patient_id'])
    pts_fold  = pts_train | pts_test

    print(f"\n--- {fold} ---")
    print(f"pacients: train={len(pts_train)}  test={len(pts_test)}  total={len(pts_fold)}")
    print(f"patches:  train={len(train):6d}  test={len(test):6d}")
    for cl in ['NC','G3','G4','G5']:
        pct_train = round(train[cl].sum()/len(train)*100,1)
        pct_test  = round(test[cl].sum()/len(test)*100,1)
        print(f"    {cl}: train={pct_train:4.1f}%  test={pct_test:4.1f}%")

# opcional: taula resumida
summary = []
for fold in ['Val1','Val2','Val3','Val4']:
    df_fold = df_all[df_all['fold'] == fold]
    for split in ['Train','Test']:
        df_s = df_fold[df_fold['split'] == split]
        pts = set(df_s['patient_id'])
        summary.append({
            'fold': fold, 'split': split,
            'n_patches': len(df_s),
            'n_patients': len(pts),
            'NC%': round(df_s['NC'].sum()/len(df_s)*100,1),
            'G3%': round(df_s['G3'].sum()/len(df_s)*100,1),
            'G4%': round(df_s['G4'].sum()/len(df_s)*100,1),
            'G5%': round(df_s['G5'].sum()/len(df_s)*100,1),
        })
summary_df = pd.DataFrame(summary)
print("\nResum en taula:")
print(summary_df)

# verificació extra: la unió dels quatre folds
unio_folds = set()
for fold in ['Val1','Val2','Val3','Val4']:
    df_fold = df_all[df_all['fold']==fold]
    unio_folds |= set(df_fold['patient_id'])
print("\nPacients únics en els quatre folds:", len(unio_folds))  # 95 també