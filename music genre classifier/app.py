import numpy as np
import librosa
import librosa.display
import gradio as gr
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import tempfile
import os
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.saving import register_keras_serializable
from convert import convert_to_wav

matplotlib.use("Agg")

# ── Custom metric (must match training) ──────────────────────────────────────
@register_keras_serializable()
def top_3_accuracy(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "genre_classifier_cnn.keras")
model = tf.keras.models.load_model(model_path)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# ── Constants (must match training) ──────────────────────────────────────────
SAMPLE_RATE   = 22050
SEGMENT_LEN   = 3          # seconds per segment
N_MFCC        = 13
N_FFT         = 2048
HOP_LENGTH    = 512
MAX_LEN       = 130        # MFCC frames per segment
TOP_K         = 3


def extract_segment_mfcc(signal: np.ndarray, sr: int) -> np.ndarray:
    """Extract and normalise a (MAX_LEN, N_MFCC) MFCC array from a signal."""
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).T  # shape: (frames, N_MFCC)

    # Pad or truncate to fixed length
    if mfcc.shape[0] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:MAX_LEN]

    # ── FIX 4: Normalise MFCCs (zero-mean, unit-variance per feature) ────────
    mean = mfcc.mean(axis=0, keepdims=True)
    std  = mfcc.std(axis=0, keepdims=True) + 1e-8
    mfcc = (mfcc - mean) / std

    return mfcc


def predict_from_upload(file_path: str):
    try:
        # ── FIX 2: Use tempfile so concurrent users don't collide ─────────────
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav = tmp.name
        spec_fd, spec_path = tempfile.mkstemp(suffix=".png")
        os.close(spec_fd)

        try:
            convert_to_wav(file_path, temp_wav)
            signal, sr = librosa.load(temp_wav, sr=SAMPLE_RATE)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        # ── Mel spectrogram (visualisation only) ─────────────────────────────
        S     = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_dB  = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH,
                                 x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
        ax.set_title('Mel Spectrogram')
        fig.tight_layout()
        fig.savefig(spec_path)
        plt.close(fig)

        # ── FIX 1: Segment-based inference (majority-vote across 3s chunks) ───
        samples_per_seg = SEGMENT_LEN * sr
        num_segments = int(max(1, len(signal) // samples_per_seg))

        vote_totals = np.zeros(len(GENRES))
        for i in range(num_segments):
            chunk = signal[i * samples_per_seg : (i + 1) * samples_per_seg]
            if len(chunk) < samples_per_seg:
                chunk = np.pad(chunk, (0, int(samples_per_seg - len(chunk))))
                
            mfcc = extract_segment_mfcc(chunk, int(sr))
            mfcc = mfcc[np.newaxis, ..., np.newaxis]   # (1, 130, 13, 1)
            probs = model.predict(mfcc, verbose=0)[0]  # (10,)
            vote_totals += probs

        avg_probs = vote_totals / num_segments

        # ── FIX 3: Top-3 predictions ──────────────────────────────────────────
        top3_idx  = np.argsort(avg_probs)[::-1][:TOP_K]
        lines     = [f"{'Genre':<12} {'Confidence':>10}",
                     "─" * 24]
        for rank, idx in enumerate(top3_idx):
            marker = " ◀" if rank == 0 else ""
            lines.append(f"{GENRES[idx]:<12} {avg_probs[idx]:>9.1%}{marker}")
        lines.append(f"\nAnalysed {num_segments} segment(s)")
        result_text = "\n".join(lines)

        return result_text, spec_path

    except Exception as e:
        return f"Prediction error: {str(e)}", None


# ── Gradio UI ─────────────────────────────────────────────────────────────────
gr.Interface(
    fn=predict_from_upload,
    inputs=gr.Audio(type="filepath", label="Upload Music Clip"),
    outputs=[
        gr.Textbox(label="Top-3 Genre Predictions", lines=7),
        gr.Image(type="pil", label="Mel Spectrogram"),
    ],
    title="🎧 Music Genre Classifier",
    description=(
        "Upload an audio clip (.mp3, .wav, .flac, .aac, .ogg, .m4a, .wma). "
        "The model splits the track into 3-second segments, predicts each one, "
        "and averages the results for a final top-3 ranking."
    ),
).launch(share=True)
