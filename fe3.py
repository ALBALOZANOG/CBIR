import os
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms
import pathlib

# === CONFIGURACIÓN ===
BASE_PATH = pathlib.Path().resolve()
IMAGES_PATH = os.path.join(BASE_PATH, 'images', 'train')
DB_PATH = os.path.join(BASE_PATH, 'database')
os.makedirs(DB_PATH, exist_ok=True)

INDEX_FILE = os.path.join(DB_PATH, 'feat_resnet.index')
CSV_FILE = os.path.join(DB_PATH, 'feat_resnet.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === MODELO ===
# Usamos ResNet50 preentrenado en ImageNet y eliminamos la última capa (clasificación)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Quitamos la capa final (fc)
model = model.to(device)
model.eval()

# === TRANSFORMACIONES ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_resnet_features(image):
    """Extrae un vector de características usando ResNet50."""
    image_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_t).cpu().numpy()
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
                feat = extract_resnet_features(img)
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

print(f"✅ Base creada con {len(features)} imágenes (ResNet50)")
print(f"Índice guardado en: {INDEX_FILE}")
print(f"CSV guardado en: {CSV_FILE}")
