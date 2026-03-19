# Music Genre Classifier

A deep learning project that classifies music into 10 genres using a Convolutional Neural Network (CNN) trained on MFCC audio features. Includes a Gradio-powered web interface for real-time genre prediction from uploaded audio files.

---

## Features

- Classifies audio into 10 genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`
- Splits audio into 3-second segments, predicts each one, and averages results for robust inference
- Displays top-3 genre predictions with confidence scores
- Renders a Mel spectrogram alongside each prediction
- Supports `.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.m4a`, `.wma`

---

## Project Structure

```
Music-Genre-Classifier/
├── music genre classifier/
│   ├── app.py               # Gradio web app
│   ├── convert.py           # Audio-to-WAV conversion via ffmpeg
│   ├── model/
│   │   └── genre_classifier_cnn.keras
│   └── notebook/
│       └── music_genre_classifier.ipynb   # Training pipeline
├── requirements.txt
└── README.md
```

---

## Model Summary

| Property | Value |
|---|---|
| Architecture | CNN (3 conv blocks) |
| Input shape | `(130, 13, 1)` — MFCC frames |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Metrics | Accuracy, Top-3 Accuracy |
| Regularisation | L2 + Dropout + BatchNorm |
| Callbacks | EarlyStopping, ReduceLROnPlateau |
| Training data | GTZAN + FMA Small (combined, class-weighted) |

---

## Datasets

> The datasets are **not included** in this repository due to size constraints.

- **[GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)** — 1,000 audio tracks × 30s across 10 genres. Place in `data/genres_original/`.
- **[FMA Small](https://github.com/mdeff/fma)** — 8,000 audio clips × 30s. Place audio in `data/fma_small/` and metadata CSV in `data/fma_metadata/tracks.csv`.

---

## Setup Instructions

### Prerequisites

- Python 3.11 (recommended — some packages don't support 3.13+ yet)
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH

### 1. Create a conda environment (recommended)

```bash
conda create -n musicgenre python=3.11 -y
conda activate musicgenre
```

### 2. Install dependencies

```bash
/path/to/anaconda3/envs/musicgenre/bin/pip install -r requirements.txt
```

> On macOS with Anaconda, use the full pip path to ensure packages install into the conda env.

### 3. Run the app

```bash
python "music genre classifier/app.py"
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## How Inference Works

1. Uploaded audio is converted to WAV via ffmpeg
2. The WAV is split into 3-second segments (matching training conditions)
3. MFCCs are extracted and normalised (zero-mean, unit-variance) for each segment
4. The CNN predicts genre probabilities for each segment
5. Probabilities are averaged across all segments
6. Top-3 genres are returned with confidence scores

---

## Demo

![Gradio UI showing jazz prediction at 100% confidence](https://github.com/user-attachments/assets/dc7e19f3-57dd-4d15-b415-91111aadf335)
