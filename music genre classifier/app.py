import numpy as np
import librosa
import gradio as gr
import tensorflow as tf
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.saving import register_keras_serializable
import os
from convert import convert_to_wav

@register_keras_serializable()
def top_3_accuracy(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)

# Load your trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "genre_classifier_cnn.keras")
model = tf.keras.models.load_model(model_path)

# Define your genre labels (must match training)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

import librosa.display
import matplotlib.pyplot as plt

def predict_from_upload(file_path):
    try:
        temp_wav = "temp.wav"
        convert_to_wav(file_path, temp_wav)

        signal, sr = librosa.load(temp_wav, sr=22050)
        
        # Generate Mel Spectrogram
        S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Save spectrogram image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        spec_img_path = "temp_spectrogram.png"
        plt.savefig(spec_img_path)
        plt.close()

        # Prepare MFCCs for prediction
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512).T
        mfcc = librosa.util.fix_length(mfcc, size=130, axis=0)
        mfcc = mfcc[..., np.newaxis]
        mfcc = np.expand_dims(mfcc, axis=0)

        # Predict genre
        prediction = model.predict(mfcc)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]
        genre = GENRES[predicted_index]

        result_text = f"Predicted Genre: {genre}\nConfidence: {confidence:.2%}"
        return result_text, spec_img_path

    except Exception as e:
        return f"Prediction error: {str(e)}", None


# Launch Gradio interface
gr.Interface(
    fn=predict_from_upload,
    inputs=gr.Audio(type="filepath", label="Upload Music Clip"),
    outputs=[
        gr.Textbox(label="Predicted Genre with Confidence"),
        gr.Image(type="pil", label="Spectrogram")
    ],
    title="🎧 Music Genre Classifier with Spectrogram"
).launch(share=True) # for public sharing