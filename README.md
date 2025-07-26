# Music Genre Classifier

A deep learning-based project that classifies music into genres using Convolutional Neural Networks (CNNs). This project includes data preprocessing, model training, and a Gradio-powered web interface for real-time genre prediction from audio files.

## Datasets Used

- **GTZAN Dataset**  
  A standard music genre classification dataset containing 1000 audio tracks, each 30 seconds long, categorized into 10 genres:
  `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.
- **FMA (Free Music Archive)**
  For dataset expansion

> Note: The datasets are **not included** in this repository due to size constraints.

## Model Summary

- **Architecture**: CNN trained on MFCC-extracted spectrograms
- **Input Shape**: `(130, 13, 1)` MFCC frames
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metric**: Accuracy, Top-3 Accuracy (custom)

## Demo (Gradio)

The interface lets you:
- Upload an audio file (`.mp3`, `.flac`, `.aac`, `.ogg`, `.m4a`, `.wma`, `.wav`)
- View the spectogram
- Get genre prediction with confidence score

https://github.com/user-attachments/assets/dc7e19f3-57dd-4d15-b415-91111aadf335

## Setup Instructions

- Install dependencies:
  ```bash
  pip install -r requirements.txt
- Run the Gradio app:
  ```bash
  python app.py
