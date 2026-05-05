# Content-Based Image Retrieval (CBIR) System

A comprehensive image retrieval system that uses multiple feature extraction methods and similarity search to find visually similar images. This project implements and compares various computer vision techniques for image feature extraction and indexing.

## Features

- **Multiple Feature Extractors:**
  - **FE1 - Color Histogram:** Color distribution-based features using 8x8x8 bins
  - **FE2 - SIFT:** Scale-Invariant Feature Transform for keypoint detection
  - **FE3 - VGG16:** Deep learning features from pre-trained VGG16 network
  - **FE4 - ResNet50:** State-of-the-art deep learning features from ResNet50
  - **FE5 - CLIP ViT-B/32:** Vision Transformer embeddings via OpenAI's CLIP model

- **Fast Similarity Search:** FAISS (Facebook AI Similarity Search) for efficient nearest neighbor retrieval
- **Interactive Web Interface:** Streamlit-based UI for image querying and visualization
- **Comprehensive Evaluation:** Performance metrics for each extraction method
- **Dataset Management:** Support for multi-category image datasets

## Dataset Structure

The project uses a dataset organized by categories:

```
images/
├── train/          # Training/Database images
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── urban/
└── test/           # Test/Query images
    ├── forest/
    ├── glacier/
    ├── mountain/
    ├── sea/
    └── urban/
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd CBIR
```

2. **Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dependencies

- `streamlit` - Web application framework
- `torch` - Deep learning framework
- `torchvision` - Computer vision models
- `faiss-cpu` - Similarity search library
- `scikit-image` - Image processing
- `pillow` - Image manipulation
- `pandas` & `numpy` - Data processing
- `clip` - OpenAI's CLIP model (vision-language)

See `requirements.txt` for complete list with versions.

## Usage

### 1. Build Feature Indexes

Run the feature extractors to create the database indexes:

```bash
# Extract histogram features
python fe1.py

# Extract SIFT features
python fe2.py

# Extract VGG16 features
python fe3.py

# Extract ResNet50 features
python fe4.py

# Extract CLIP features
python fe5.py
```

Each script generates:
- A FAISS index file (`.index`)
- A CSV file with feature vectors and metadata

### 2. Run the Web Application

Launch the interactive Streamlit interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Search for Similar Images

1. Upload or select an image
2. Choose a feature extraction method
3. Set the number of results to retrieve
4. View similar images ranked by similarity score

## Project Structure

```
database/
├── feat_hist.csv / feat_hist.index      # Histogram features
├── feat_sift.csv / feat_sift.index      # SIFT features
├── feat_vgg.csv / feat_vgg.index        # VGG16 features
├── feat_resnet.csv / feat_resnet.index  # ResNet50 features
└── feat_clip.csv / feat_clip.index      # CLIP features

comparisons/
├── comparisons.csv                      # Comparison results
└── evals/                              # Evaluation metrics
    ├── extractor_1_histograma.json
    ├── extractor_2_sift.json
    ├── extractor_3_vgg16.json
    ├── extractor_4_resnet50.json
    └── extractor_5_clip_vit_b_32.json
```

## Feature Extractors Overview

| Extractor | Method | Dimensions | Speed | Accuracy |
|-----------|--------|-----------|-------|----------|
| FE1 | Color Histogram | 512 | Very Fast | Low |
| FE2 | SIFT Keypoints | Variable | Moderate | Moderate |
| FE3 | VGG16 CNN | 4096 | Moderate | High |
| FE4 | ResNet50 CNN | 2048 | Moderate | Very High |
| FE5 | CLIP ViT-B/32 | 512 | Moderate | Very High |

## Evaluation

Evaluation metrics are stored in JSON format for each extractor:

```bash
comparisons/evals/extractor_N_*.json
```

These files contain:
- Precision and Recall metrics
- Mean Average Precision (mAP)
- Performance benchmarks

## Interactive Features in Streamlit App

- **Image Cropping Tool:** Crop uploaded images before searching
- **Multiple Query Modes:** Test mode for query images or upload custom images
- **Visual Results:** Display top-k similar images with similarity scores
- **Comparison View:** Compare results across different feature extractors
- **Performance Metrics:** View detailed evaluation statistics

## How It Works

1. **Feature Extraction:** Images are processed using selected feature extraction method
2. **Indexing:** Features are indexed using FAISS for fast retrieval
3. **Query Processing:** Query image features are extracted using the same method
4. **Similarity Search:** FAISS finds k-nearest neighbors in the feature space
5. **Result Ranking:** Similar images are ranked by similarity score and displayed

## Configuration

Edit the following in the source files to customize:

- **Histogram bins:** Modify `bins=(8, 8, 8)` in feature extractors
- **Number of results:** Adjust `k` parameter in Streamlit app
- **Model architectures:** Change pre-trained models in feature extraction scripts
- **Dataset paths:** Update path variables in each extractor file

## Notes

- CLIP model requires internet connection for first download
- Deep learning models (VGG16, ResNet50) require GPU for optimal performance (CPU supported but slower)
- SIFT features are variable-dimensional and are aggregated for indexing
- All feature vectors are normalized for cosine similarity search

## Dataset

This project uses the **Intel Image Classification** dataset from Kaggle:
- **Source:** [Intel Image Classification - Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Categories:** Forest, Glacier, Mountain, Sea, Urban
- **Split:** Training and test sets included

## Academic Context

This project was developed as part of the **Algorithms and Architecture for Image Processing** (*Algoritmos y Arquitectura para el Procesamiento de Imágenes*) course at the University, implementing and comparing various image retrieval techniques for content-based image search systems.

## License

No license specified.

## Authors

- **Alba Lozano Guixa**
- **Javier González Sorli**

## Contributing

Contributions are welcome! Feel free to submit issues and enhancement requests.

---

**Last Updated:** May 2026
