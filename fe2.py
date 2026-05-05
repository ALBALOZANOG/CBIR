import os
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import pathlib
from skimage import color
from skimage.feature import SIFT

# === CONFIGURACIÓN ===
BASE_PATH = pathlib.Path().resolve()
IMAGES_PATH = os.path.join(BASE_PATH, 'images', 'train')
DB_PATH = os.path.join(BASE_PATH, 'database')
os.makedirs(DB_PATH, exist_ok=True)

INDEX_FILE = os.path.join(DB_PATH, 'feat_sift.index')
CSV_FILE = os.path.join(DB_PATH, 'feat_sift.csv')

# === FUNCIÓN DE EXTRACCIÓN ===
def extract_sift_features(image):
    """
    Extrae un vector fijo usando SIFT.
    Resumimos los descriptores con media y desviación estándar (256 dimensiones).
    """
    gray = color.rgb2gray(np.array(image.convert("RGB")))
    extractor = SIFT()
    extractor.detect_and_extract(gray)
    desc = extractor.descriptors  # (n_keypoints, 128)

    if desc is None or len(desc) == 0:
        return np.zeros(256, dtype='float32')

    desc_mean = desc.mean(axis=0)
    desc_std = desc.std(axis=0)
    feat = np.concatenate([desc_mean, desc_std]).astype('float32')
    return feat


# === PROCESAR TODAS LAS IMÁGENES ===
features = []
image_paths = []
labels = []

for root, _, files in os.walk(IMAGES_PATH):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(root, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                feat = extract_sift_features(img)
                features.append(feat)
                image_paths.append(os.path.relpath(img_path, IMAGES_PATH))
                labels.append(os.path.basename(os.path.dirname(img_path)))
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

print(f"✅ Base creada con {len(features)} imágenes (SIFT)")
print(f"Índice guardado en: {INDEX_FILE}")
print(f"CSV guardado en: {CSV_FILE}")
