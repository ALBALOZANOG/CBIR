import os
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import pathlib

# === CONFIGURACIÓN DE RUTAS ===
BASE_PATH = pathlib.Path().resolve()
IMAGES_PATH = os.path.join(BASE_PATH, 'images', 'train')  # usamos train para crear la base
DB_PATH = os.path.join(BASE_PATH, 'database')
os.makedirs(DB_PATH, exist_ok=True)

# === ARCHIVOS DE SALIDA ===
INDEX_FILE = os.path.join(DB_PATH, 'feat_hist.index')
CSV_FILE = os.path.join(DB_PATH, 'feat_hist.csv')


def extract_histogram_features(image, bins=(8, 8, 8)):
    """Convierte una imagen en un vector de características basado en histogramas de color."""
    image = image.convert("RGB")
    hist = np.histogramdd(
        np.array(image).reshape(-1, 3),
        bins=bins,
        range=((0, 256), (0, 256), (0, 256))
    )[0]
    hist = hist / np.sum(hist)
    return hist.flatten().astype('float32')


# === PROCESAR TODAS LAS IMÁGENES DE FORMA RECURSIVA ===
features = []
image_paths = []
labels = []

for root, _, files in os.walk(IMAGES_PATH):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(root, filename)
            try:
                img = Image.open(img_path)
                feat = extract_histogram_features(img)
                features.append(feat)
                image_paths.append(os.path.relpath(img_path, IMAGES_PATH))  # ruta relativa
                label = os.path.basename(os.path.dirname(img_path))  # nombre de carpeta = clase
                labels.append(label)
            except Exception as e:
                print(f"⚠️ Error con {img_path}: {e}")

features = np.array(features).astype('float32')

# === CREAR ÍNDICE FAISS ===
d = features.shape[1]
index = faiss.IndexFlatL2(d)
faiss.normalize_L2(features)
index.add(features)

# === GUARDAR ÍNDICE Y CSV ===
faiss.write_index(index, INDEX_FILE)
pd.DataFrame({'image': image_paths, 'label': labels}).to_csv(CSV_FILE, index=False)

print(f"✅ Base creada con {len(features)} imágenes")
print(f"Índice guardado en: {INDEX_FILE}")
print(f"CSV guardado en: {CSV_FILE}")
