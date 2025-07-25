# Music Genre Classifier

A deep learning-based project that classifies music into genres using audio spectrograms and Convolutional Neural Networks (CNNs). This project includes data preprocessing, model training, and a Gradio-powered web interface for real-time genre prediction from audio files.

## Datasets Used

- **GTZAN Dataset**  
  A standard music genre classification dataset containing 1000 audio tracks, each 30 seconds long, categorized into 10 genres:
  `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.

> Note: The dataset is **not included** in this repository due to size constraints.

## Model Summary

- **Architecture**: CNN trained on MFCC-extracted spectrograms
- **Input Shape**: `(130, 13, 1)` MFCC frames
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metric**: Accuracy, Top-3 Accuracy (custom)

## Demo (Gradio)

Run the app to test predictions on your own audio files:
```bash
python app.py
