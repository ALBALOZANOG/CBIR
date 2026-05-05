import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import clip
import json
import re
from datetime import datetime
from collections import Counter
from skimage import color
from skimage.feature import SIFT

import streamlit as st
from streamlit_cropper import st_cropper
from torchvision import models, transforms

# ======================
# CONFIGURACIÓN INICIAL
# ======================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(layout="wide")

# Rutas principales y carpetas de persistencia
FILES_PATH = str(pathlib.Path().resolve())
IMAGES_PATH = os.path.join(FILES_PATH, 'images', 'train')
TEST_PATH = os.path.join(FILES_PATH, 'images', 'test')
DB_PATH = os.path.join(FILES_PATH, 'database')
COMPARISONS_PATH = os.path.join(FILES_PATH, 'comparisons')
EVALS_PATH = os.path.join(COMPARISONS_PATH, 'evals')
os.makedirs(EVALS_PATH, exist_ok=True)

# Validar que las carpetas existen
if not os.path.exists(IMAGES_PATH):
    st.error(f"❌ No se encontró la carpeta de imágenes: {IMAGES_PATH}")
    st.stop()

if not os.path.exists(TEST_PATH):
    st.warning(f"⚠️ No existe carpeta de test: {TEST_PATH}")

if not os.path.exists(DB_PATH):
    st.error(f"❌ No se encontró la base de datos: {DB_PATH}")
    st.stop()


# ======================
# FUNCIONES AUXILIARES (extractores y evaluaciones)
# ======================

# ----- Extractores de características -----

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


def extract_clip_features(image):
    """Extrae embedding de imagen con CLIP ViT-B/32."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    image_t = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_t).float().cpu().numpy()
    return features.flatten().astype('float32')


def extract_vgg_features(image):
    """Extrae características profundas de una imagen usando VGG16 preentrenado."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_t).cpu().numpy()

    return features.flatten().astype('float32')


def extract_resnet_features(image):
    """Extrae características de una imagen usando ResNet50."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_t).cpu().numpy()

    return features.flatten().astype('float32')


def get_image_list(csv_name):
    """Lee el CSV de base de datos (imagen + label)."""
    df = pd.read_csv(os.path.join(DB_PATH, csv_name))
    return list(df.image.values), list(df.label.values)


# ----- Recuperación puntual (búsqueda manual) -----
def retrieve_image(img_query, feature_extractor, n_imgs=9):
    """Busca las imágenes más similares según el extractor seleccionado."""
    if feature_extractor == 'Extractor 1 (Histograma)':
        vector = extract_histogram_features(img_query)
        vector = np.expand_dims(vector, axis=0)
        faiss.normalize_L2(vector)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_hist.index'))
        csv_name = 'feat_hist.csv'

    elif feature_extractor == 'Extractor 2 (SIFT)':
        vector = extract_sift_features(img_query)
        vector = np.expand_dims(vector, axis=0)
        faiss.normalize_L2(vector)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_sift.index'))
        csv_name = 'feat_sift.csv'

    elif feature_extractor == 'Extractor 3 (VGG16)':
        vector = extract_vgg_features(img_query)
        vector = np.expand_dims(vector, axis=0)
        faiss.normalize_L2(vector)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_vgg.index'))
        csv_name = 'feat_vgg.csv'

    elif feature_extractor == 'Extractor 4 (ResNet50)':
        vector = extract_resnet_features(img_query)
        vector = np.expand_dims(vector, axis=0)
        faiss.normalize_L2(vector)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_resnet.index'))
        csv_name = 'feat_resnet.csv'

    elif feature_extractor == 'Extractor 5 (CLIP ViT-B/32)':
        vector = extract_clip_features(img_query)
        vector = np.expand_dims(vector, axis=0)
        faiss.normalize_L2(vector)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_clip.index'))
        csv_name = 'feat_clip.csv'

    else:
        st.error("Extractor no implementado aún.")
        return [], None

    _, indices = indexer.search(vector, k=n_imgs)
    return indices[0], csv_name


EXTRACTOR_CONFIG = {
    'Extractor 1 (Histograma)': {
        'index': 'feat_hist.index',
        'csv': 'feat_hist.csv',
        'fn': extract_histogram_features
    },
    'Extractor 2 (SIFT)': {
        'index': 'feat_sift.index',
        'csv': 'feat_sift.csv',
        'fn': extract_sift_features
    },
    'Extractor 3 (VGG16)': {
        'index': 'feat_vgg.index',
        'csv': 'feat_vgg.csv',
        'fn': extract_vgg_features
    },
    'Extractor 4 (ResNet50)': {
        'index': 'feat_resnet.index',
        'csv': 'feat_resnet.csv',
        'fn': extract_resnet_features
    },
    'Extractor 5 (CLIP ViT-B/32)': {
        'index': 'feat_clip.index',
        'csv': 'feat_clip.csv',
        'fn': extract_clip_features
    },
}


# ----- Métricas -----

def average_precision_at_k(hits, relevant_total):
    """AP@k: hits es lista de 1/0 en el top-k."""
    if relevant_total == 0:
        return 0.0
    precisions = []
    hit_count = 0
    for i, h in enumerate(hits, start=1):
        if h:
            hit_count += 1
            precisions.append(hit_count / i)
    if not precisions:
        return 0.0
    return sum(precisions) / min(relevant_total, len(hits))


def compute_metrics_and_failures(queries, k, label_counts, avg_time_ms=None):
    """Calcula métricas y fallos para un valor de k usando resultados ya obtenidos."""
    total_queries = len(queries)
    if total_queries == 0:
        return None, []

    top1 = 0.0
    precision_at_k = 0.0
    recall_at_k = 0.0
    map_at_k = 0.0
    failures = []

    for q in queries:
        true_label = q['query_label']
        retrieved_labels = q['retrieved_labels'][:k]
        hits = [1 if lbl == true_label else 0 for lbl in retrieved_labels]
        hits_sum = sum(hits)

        top1 += hits[0]
        precision_at_k += hits_sum / k
        recall_at_k += hits_sum / max(1, label_counts[true_label])
        map_at_k += average_precision_at_k(hits, label_counts[true_label])

        try:
            first_fail_idx = hits.index(0)
            failures.append({
                'query_path': q['query_path'],
                'query_label': true_label,
                'wrong_label': retrieved_labels[first_fail_idx],
                'rank': first_fail_idx + 1,
                'wrong_path': q['retrieved_paths'][first_fail_idx]
            })
        except ValueError:
            # no fail in top-k
            pass

    metrics = {
        'top1': top1 / total_queries,
        'p_at_k': precision_at_k / total_queries,
        'recall_at_k': recall_at_k / total_queries,
        'map_at_k': map_at_k / total_queries,
        'avg_time_ms': avg_time_ms if avg_time_ms is not None else 0.0,
        'queries': total_queries
    }
    return metrics, failures


# ----- Evaluación de Test -----

def evaluate_extractor(feature_extractor, k=5, top_n=10):
    """Evalúa un extractor sobre el split de test y guarda top_n resultados para reuso."""
    config = EXTRACTOR_CONFIG.get(feature_extractor)
    if not config:
        return None, [], []

    index_path = os.path.join(DB_PATH, config['index'])
    csv_path = os.path.join(DB_PATH, config['csv'])

    if not os.path.exists(index_path) or not os.path.exists(csv_path):
        st.error(f"Falta índice o CSV para {feature_extractor}. Ejecuta el script correspondiente.")
        return None, [], [], []

    indexer = faiss.read_index(index_path)
    df = pd.read_csv(csv_path)
    
    # Limpiar rutas en el CSV (convertir rutas antiguas a relativas si es necesario)
    image_list = []
    for img_path in df.image.values:
        # Si contiene CDIA_oficial u otra ruta antigua, extraer solo la parte relativa
        if 'CDIA_oficial' in str(img_path) or '\\' in str(img_path):
            # Extraer solo la parte desde images/train
            parts = str(img_path).split(os.sep)
            if 'images' in parts:
                idx = parts.index('images')
                # Reconstruir como ruta relativa
                rel_path = os.sep.join(parts[idx+1:])
                image_list.append(rel_path)
            else:
                image_list.append(str(img_path))
        else:
            image_list.append(str(img_path))
    
    label_list = list(df.label.values)
    label_counts = Counter(label_list)

    fn = config['fn']

    total_time = 0.0
    queries = []

    n_search = max(k, top_n)

    for root, _, files in os.walk(TEST_PATH):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(root, filename)
            true_label = os.path.basename(os.path.dirname(img_path))

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                st.warning(f"No se pudo abrir la imagen de test: {img_path} ({e})")
                continue

            start = time.time()
            vector = fn(img)
            vector = np.expand_dims(vector, axis=0)
            faiss.normalize_L2(vector)
            _, idxs = indexer.search(vector, k=n_search)
            total_time += (time.time() - start)

            retrieved_idxs = idxs[0].tolist()
            retrieved_labels = [label_list[i] for i in retrieved_idxs]
            # Construir rutas correctamente: reemplazar separadores Windows con OS
            retrieved_paths = [os.path.join(IMAGES_PATH, image_list[i].replace('\\', os.sep).replace('/', os.sep)) for i in retrieved_idxs]

            queries.append({
                'query_path': img_path,
                'query_label': true_label,
                'retrieved_labels': retrieved_labels,
                'retrieved_paths': retrieved_paths
            })

    if len(queries) == 0:
        st.warning("No se encontraron imágenes en la carpeta de test.")
        return None, [], []

    metrics, failures = compute_metrics_and_failures(queries, k, label_counts)
    if metrics:
        metrics['avg_time_ms'] = (total_time / len(queries)) * 1000

    return metrics, failures, {'queries': queries, 'label_counts': dict(label_counts)}


# ----- Persistencia de evaluaciones (leer/guardar JSON) -----

def _slugify_extractor(name):
    """Convierte el nombre del extractor en un slug seguro para archivo."""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name).strip('_')
    return name


def save_eval_data(extractor, k, metrics, failures):
    """Guarda resultados de evaluación (métricas + fallos + queries) por extractor en un único JSON."""
    if not metrics:
        return
    fname = f"{_slugify_extractor(extractor)}.json"
    path = os.path.join(EVALS_PATH, fname)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    if 'queries_data' not in existing or not existing.get('queries_data'):
        existing['queries_data'] = {}
    if 'evaluations' not in existing or not isinstance(existing.get('evaluations'), dict):
        existing['evaluations'] = {}

    # Persist queries/label_counts once (son independientes de k si top_n >= k max)
    if 'queries' in failures:  # never true, fallback safeguard
        pass

    payload_queries = st.session_state.get('last_queries_payload')
    if payload_queries:
        existing['queries_data'] = payload_queries

    existing['evaluations'][str(k)] = {
        'metrics': metrics,
        'failures': failures,
        'timestamp': datetime.now().isoformat()
    }

    payload = {
        'extractor': extractor,
        'queries_data': existing.get('queries_data', {}),
        'evaluations': existing['evaluations']
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_eval_data(extractor):
    """Carga datos guardados de evaluación para un extractor (incluye compatibilidad legacy)."""
    fname = f"{_slugify_extractor(extractor)}.json"
    path = os.path.join(EVALS_PATH, fname)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                'evaluations': data.get('evaluations', {}),
                'queries_data': data.get('queries_data')
            }
        except Exception:
            pass

    # Compatibilidad con archivos legacy por K (ej: extractor_1_histograma_k3.json)
    legacy_files = [f for f in os.listdir(EVALS_PATH) if f.startswith(_slugify_extractor(extractor)) and '_k' in f]
    evaluations = {}
    for lf in legacy_files:
        k_match = re.search(r'_k(\d+)', lf)
        if not k_match:
            continue
        lk = k_match.group(1)
        legacy_path = os.path.join(EVALS_PATH, lf)
        try:
            with open(legacy_path, 'r', encoding='utf-8') as f:
                legacy = json.load(f)
            metrics = legacy.get('metrics', {})
            failures = legacy.get('failures', [])
            if metrics:
                evaluations[lk] = {'metrics': metrics, 'failures': failures, 'timestamp': legacy.get('timestamp')}
        except Exception:
            continue
    if evaluations:
        payload = {'evaluations': evaluations, 'queries_data': None}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload
    return None


# ======================
# INTERFAZ STREAMLIT
# ======================

def main():
    st.title('🔎 CBIR - Image Search (Content-Based Image Retrieval)')

    # Tabs principales: búsqueda en vivo, evaluación en test y comparativa
    tab_search, tab_eval, tab_compare = st.tabs(["🔍 Búsqueda", "📊 Evaluación", "📈 Comparativa"])

    with tab_search:
        col1, col2 = st.columns(2)

        # --- Panel de consulta y recorte ---
        with col1:
            st.header('QUERY')

            option = st.selectbox(
                'Selecciona el extractor:',
                (
                    'Extractor 1 (Histograma)',
                    'Extractor 2 (SIFT)',
                    'Extractor 3 (VGG16)',
                    'Extractor 4 (ResNet50)',
                    'Extractor 5 (CLIP ViT-B/32)',
                )
            )

            img_file = st.file_uploader('Sube una imagen de consulta', type=['png', 'jpg', 'jpeg'])

            if img_file:
                img = Image.open(img_file)
                cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
                st.write("Vista previa (recorte)")
                _ = cropped_img.thumbnail((150, 150))
                st.image(cropped_img)

        # --- Panel de resultados de búsqueda ---
        with col2:
            st.header('RESULTADOS')

            if img_file:
                st.markdown('**Buscando imágenes similares...**')
                start = time.time()

                retriev, csv_name = retrieve_image(cropped_img, option, n_imgs=9)
                if retriev is None or csv_name is None:
                    return

                image_list, label_list = get_image_list(csv_name)

                end = time.time()
                st.markdown(f'**Finalizado en {end - start:.2f} segundos**')

                cols = st.columns(3)
                for i, idx in enumerate(retriev):
                    img_path = os.path.join(IMAGES_PATH, image_list[idx])
                    label = label_list[idx]

                    try:
                        img = Image.open(img_path)
                        cols[i % 3].image(img, use_column_width=True, caption=f"Clase: {label}")
                    except Exception as e:
                        st.warning(f"No se pudo abrir la imagen: {img_path} ({e})")

    with tab_eval:
        st.header("Evaluación en test")
        option_eval = st.selectbox(
            'Extractor a evaluar:',
            (
                'Extractor 1 (Histograma)',
                'Extractor 2 (SIFT)',
                'Extractor 3 (VGG16)',
                'Extractor 4 (ResNet50)',
                'Extractor 5 (CLIP ViT-B/32)',
            ),
            key="eval_extractor"
        )
        k = st.selectbox('Valor de K para Precision@K / mAP@K', options=[3, 5, 7], index=1, key="k_eval")

        def render_eval(metrics, failures):
            st.subheader("Métricas globales")
            colm = st.columns(5)
            colm[0].metric("Top-1 acc", f"{metrics['top1']*100:.1f}%")
            colm[1].metric(f"Precision@{k}", f"{metrics['p_at_k']*100:.1f}%")
            colm[2].metric(f"Recall@{k}", f"{metrics['recall_at_k']*100:.1f}%")
            colm[3].metric(f"mAP@{k}", f"{metrics['map_at_k']*100:.1f}%")
            colm[4].metric("Tiempo medio (ms)", f"{metrics['avg_time_ms']:.1f}")
            st.caption(f"Queries evaluadas: {metrics['queries']}")

            if failures:
                st.subheader("Fallos (primer error en el top-K)")
                fail_rows = []
                for f in failures:
                    fail_rows.append({
                        'Query': os.path.basename(f['query_path']),
                        'Clase real': f['query_label'],
                        'Clase devuelta': f['wrong_label'],
                        'Posición': f['rank']
                    })
                st.dataframe(pd.DataFrame(fail_rows), use_container_width=True)

                # Agregados por clase real
                total_fail = len(failures)
                per_class = Counter([f['query_label'] for f in failures])
                per_pair = {}
                for f in failures:
                    per_pair.setdefault(f['query_label'], Counter())
                    per_pair[f['query_label']][f['wrong_label']] += 1

                agg_rows = []
                for cls, cnt in per_class.items():
                    conc = (cnt / total_fail) * 100 if total_fail else 0.0
                    wrong_counter = per_pair.get(cls, Counter())
                    if wrong_counter:
                        max_count = max(wrong_counter.values())
                        top_labels = [l for l, c in wrong_counter.items() if c == max_count]
                        top_labels.sort()
                        label_str = " / ".join(top_labels)
                        freq = (max_count / cnt) * 100 if cnt else 0.0
                        freq_str = f"{freq:.1f}%"
                        if len(top_labels) > 1:
                            freq_str += " c/u"
                    else:
                        label_str = "N/A"
                        freq_str = "N/A"
                    agg_rows.append({
                        'Clase real': cls,
                        'Conc. de error': f"{conc:.1f}%",
                        'Clase errónea más frecuente': label_str,
                        'Frec. de misetiquetado': freq_str
                    })

                if agg_rows:
                    st.subheader("Agregados de fallos (primer error)")
                    st.dataframe(pd.DataFrame(agg_rows), use_container_width=True)

                st.subheader("Visualización de fallos")
                per_row = 4  # 4 fallos -> 8 imágenes por fila (pares)
                for i in range(0, len(failures), per_row):
                    row_fails = failures[i:i+per_row]
                    widths = []
                    for _ in row_fails:
                        widths.extend([1, 1, 0.2])
                    if widths:
                        widths = widths[:-1]  # quitar último espaciador
                    cols = st.columns(widths)
                    for j, f in enumerate(row_fails):
                        base = j * 3
                        q_col = cols[base]
                        r_col = cols[base + 1]
                        try:
                            q_col.image(Image.open(f['query_path']), use_column_width=True)
                            q_col.markdown(
                                f"<span style='color:green; font-weight:bold'>{f['query_label']}</span>",
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            q_col.write(f"No se pudo abrir query ({e})")
                        try:
                            r_col.image(Image.open(f['wrong_path']), use_column_width=True)
                            r_col.markdown(
                                f"<span style='color:red; font-weight:bold'>{f['wrong_label']} (pos {f['rank']})</span>",
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            r_col.write(f"No se pudo abrir recuperada ({e})")
            else:
                st.info("Sin fallos en el top-K para este extractor.")

        saved_data = load_eval_data(option_eval)
        if saved_data:
            evals = saved_data.get('evaluations', {})
            queries_data = saved_data.get('queries_data')
            baseline_time = None
            for v in evals.values():
                mt = v.get('metrics', {})
                if mt and 'avg_time_ms' in mt:
                    baseline_time = mt['avg_time_ms']
                    break
            metrics_loaded = None
            failures_loaded = []
            if queries_data:
                recomputed_metrics, recomputed_failures = compute_metrics_and_failures(
                    queries_data.get('queries', []),
                    k,
                    queries_data.get('label_counts', {}),
                    avg_time_ms=baseline_time
                )
                metrics_loaded = recomputed_metrics
                failures_loaded = recomputed_failures if recomputed_metrics else []
            if metrics_loaded is None:
                stored = evals.get(str(k), {})
                metrics_loaded = stored.get('metrics')
                failures_loaded = stored.get('failures', [])
            if metrics_loaded:
                st.info("Mostrando resultados guardados.")
                render_eval(metrics_loaded, failures_loaded)

        # Botón de ejecución
        if st.button('Ejecutar evaluación', key="run_eval"):
            with st.spinner('Evaluando...'):
                metrics, failures, queries_data = evaluate_extractor(option_eval, k=k)
                st.session_state['last_queries_payload'] = queries_data
                if metrics:
                    save_eval_data(option_eval, k, metrics, failures)

            if metrics:
                render_eval(metrics, failures)

    # --- Panel de comparativa de extractores ---
    with tab_compare:
        st.header("Comparativa de extractores")
        k_all = st.selectbox('Valor de K para comparar', options=[3, 5, 7], index=1, key="k_compare")

        # Se refresca desde evaluaciones guardadas para no recalcular si ya existe el top-10 de test
        def metrics_from_saved(extractor, k):
            data = load_eval_data(extractor)
            if not data:
                return None
            evals = data.get('evaluations', {})
            queries_data = data.get('queries_data')
            baseline_time = None
            for v in evals.values():
                mt = v.get('metrics', {})
                if mt and 'avg_time_ms' in mt:
                    baseline_time = mt['avg_time_ms']
                    break
            metrics = None
            failures = []
            if queries_data:
                recomputed_metrics, recomputed_failures = compute_metrics_and_failures(
                    queries_data.get('queries', []),
                    k,
                    queries_data.get('label_counts', {}),
                    avg_time_ms=baseline_time
                )
                metrics = recomputed_metrics
                failures = recomputed_failures if recomputed_metrics else []
            if metrics is None:
                stored = evals.get(str(k), {})
                metrics = stored.get('metrics')
                failures = stored.get('failures', [])
            return metrics, failures, queries_data

        if st.button('Evaluar extractores restantes', key="run_missing_compare"):
            with st.spinner('Evaluando faltantes...'):
                for name in EXTRACTOR_CONFIG.keys():
                    data = load_eval_data(name)
                    if data and data.get('queries_data'):
                        continue
                    metrics, failures, queries_data = evaluate_extractor(name, k=k_all)
                    if metrics:
                        st.session_state['last_queries_payload'] = queries_data
                        save_eval_data(name, k_all, metrics, failures)
            st.success("Comparativa actualizada.")

        rows = []
        for name in EXTRACTOR_CONFIG.keys():
            m = metrics_from_saved(name, k_all)
            if not m:
                continue
            metrics, _, _ = m
            if not metrics:
                continue
            rows.append({
                'Extractor': name,
                'Top-1 (%)': round(metrics['top1']*100, 2),
                f'P@{k_all} (%)': round(metrics['p_at_k']*100, 2),
                f'mAP@{k_all} (%)': round(metrics['map_at_k']*100, 2),
                'Tiempo (ms)': round(metrics['avg_time_ms'], 1)
            })

        if rows:
            # Tabla agregada de rendimiento por extractor
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No hay resultados guardados aún. Ejecuta evaluación en la pestaña anterior o pulsa 'Evaluar extractores restantes'.")


if __name__ == '__main__':
    main()
