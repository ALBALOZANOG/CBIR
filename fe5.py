import os
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import torch
import pathlib
import clip  # requiere: pip install git+https://github.com/openai/CLIP.git

# === CONFIGURACIÓN ===
BASE_PATH = pathlib.Path().resolve()
IMAGES_PATH = os.path.join(BASE_PATH, 'images', 'train')
DB_PATH = os.path.join(BASE_PATH, 'database')
os.makedirs(DB_PATH, exist_ok=True)

INDEX_FILE = os.path.join(DB_PATH, 'feat_clip.index')
CSV_FILE = os.path.join(DB_PATH, 'feat_clip.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === MODELO CLIP (ViT-B/32) ===
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


def extract_clip_features(image):
    """Extrae embedding de imagen con CLIP ViT-B/32."""
    image_t = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_t).float().cpu().numpy()
    return features.flatten().astype('float32')


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
                feat = extract_clip_features(img)
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

print(f"✅ Base creada con {len(features)} imágenes (CLIP ViT-B/32)")
print(f"Índice guardado en: {INDEX_FILE}")
print(f"CSV guardado en: {CSV_FILE}")
