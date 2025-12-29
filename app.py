# =========================================
# FORCE CPU ONLY + SILENCE GPU LOGS
# =========================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================================
# FIX MATPLOTLIB (NO TKINTER IN WSL)
# =========================================
import matplotlib
matplotlib.use("Agg")

import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import librosa
import soundfile
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text as text 
import keras
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================
# CONFIGURATION & UNIFIED 7 LABELS
# =========================================
'''os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
matplotlib.use("Agg")
warnings.filterwarnings("ignore")'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PLOT_FOLDER = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# THE UNIFIED 7 EMOTIONS ORDER
UNIFIED_7 = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# =========================================
# LOAD MODELS
# =========================================
# 1. TEXT
with open(os.path.join(BASE_DIR, "models", "text", "label_encoder.pkl"), "rb") as f:
    text_label_encoder = pickle.load(f)
TEXT_ORIGINAL_LABELS = list(text_label_encoder.classes_) # Get original training order

text_model = keras.layers.TFSMLayer(
    os.path.join(BASE_DIR, "models", "text", "emotion_bert_model"),
    call_endpoint="serving_default"
)

# 2. IMAGE
image_model = load_model(os.path.join(BASE_DIR, "models", "image", "fer2013_emotion_model_final.h5"))
IMAGE_ORIGINAL_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 3. AUDIO
audio_model = load_model(os.path.join(BASE_DIR, "models", "audio", "voice_emotion_cnn_bilstm.h5"))
AUDIO_ORIGINAL_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful"]

app = Flask(__name__)

# =========================================
# MAPPING ENGINE
# =========================================
def map_to_unified_7(raw_probs, original_labels):
    """Maps any model output array to the UNIFIED_7 array."""
    # Create a dictionary to hold probabilities for our target 7 classes
    mapping = {emo: 0.0 for emo in UNIFIED_7}
    
    for i, prob in enumerate(raw_probs):
        label = str(original_labels[i]).capitalize()
        
        # Handle specific cases
        if label == "Fearful": label = "Fear"
        if label == "Calm": label = "Neutral" # Merge Calm into Neutral
        
        if label in mapping:
            mapping[label] += prob
            
    # Convert dict back to list following UNIFIED_7 order
    unified_probs = [mapping[emo] for emo in UNIFIED_7]
    
    # Re-normalize so it sums to 1.0 (100%)
    total = sum(unified_probs)
    return np.array(unified_probs) / total if total > 0 else np.array(unified_probs)

def plot_probabilities(probabilities):
    plt.figure(figsize=(8, 4))
    plt.bar(UNIFIED_7, probabilities, color='#6366f1')
    plt.ylim(0, 1)
    plt.title("Emotion Probability Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "emotion_plot.png"))
    plt.close()

# =========================================
# MAIN ROUTE
# =========================================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, plot_url = None, None, None

    if request.method == "POST":
        mode = request.form.get("mode", "").lower()

        if mode == "text":
            text_input = request.form.get("text", "").strip()
            if text_input:
                out = text_model(np.array([text_input], dtype=str), training=False)
                raw = out["classifier"].numpy()[0]
                final_probs = map_to_unified_7(raw, TEXT_ORIGINAL_LABELS)

        elif mode == "image":
            file = request.files.get("file")
            if file and file.filename != "":
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                img = image.load_img(file_path, target_size=(150, 150), color_mode="rgb")
                img_arr = image.img_to_array(img) / 255.0
                raw = image_model.predict(np.expand_dims(img_arr, axis=0))[0]
                final_probs = map_to_unified_7(raw, IMAGE_ORIGINAL_LABELS)

        elif mode == "audio":
            file = request.files.get("file")
            if file and file.filename != "":
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                X, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
                feat = pad_sequences([mfccs.T], maxlen=200, padding="post", dtype="float32")
                raw = audio_model.predict(feat)[0]
                final_probs = map_to_unified_7(raw, AUDIO_ORIGINAL_LABELS)

        # FINAL PROCESSING (Common for all)
        if 'final_probs' in locals():
            idx = np.argmax(final_probs)
            prediction = UNIFIED_7[idx]
            confidence = f"{final_probs[idx] * 100:.2f}%"
            plot_probabilities(final_probs)
            plot_url = f"/static/plots/emotion_plot.png?v={int(time.time())}"

    return render_template("index.html", prediction=prediction, confidence=confidence, plot=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
