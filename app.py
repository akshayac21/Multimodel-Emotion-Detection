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

# =========================================
# STANDARD IMPORTS
# =========================================
import numpy as np
import pickle
import librosa
import soundfile
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_text as text
import keras
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================
# GLOBAL EMOTION SCHEMA (FINAL)
# =========================================
FINAL_EMOTIONS = [
    "angry",
    "sad",
    "happy",
    "neutral",
    "surprise",
    "fear",
    "disgust"
]

# =========================================
# PATHS & FOLDERS
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PLOT_FOLDER = os.path.join(BASE_DIR, "static", "plots")
PLOT_PATH = os.path.join(PLOT_FOLDER, "emotion_plot.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# =========================================
# EMOTION HISTORY (PER MODALITY)
# =========================================
emotion_history = {
    "text": [],
    "image": [],
    "audio": []
}
MAX_HISTORY = 10

# =========================================
# LOAD TEXT MODEL
# =========================================
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text", "emotion_bert_model")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "text", "label_encoder.pkl")

with open(ENCODER_PATH, "rb") as f:
    text_label_encoder = pickle.load(f)

text_model = keras.layers.TFSMLayer(
    TEXT_MODEL_PATH,
    call_endpoint="serving_default"
)

TEXT_EMOTIONS = list(text_label_encoder.classes_)

# =========================================
# LOAD IMAGE MODEL
# =========================================
IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "image", "fer2013_emotion_model_final.h5"
)

IMAGE_CLASSES = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

IMG_SIZE = (150, 150)
image_model = load_model(IMAGE_MODEL_PATH)

# =========================================
# LOAD AUDIO MODEL
# =========================================
AUDIO_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "audio", "voice_emotion_cnn_bilstm.h5"
)

AUDIO_ORIGINAL_LABELS = [
    "neutral", "calm", "happy",
    "sad", "angry", "fearful"
]

MAX_SEQUENCE_LENGTH = 200
audio_model = load_model(AUDIO_MODEL_PATH)

# =========================================
# FLASK APP
# =========================================
app = Flask(__name__)

# =========================================
# PLOT FUNCTION
# =========================================
def plot_probabilities(labels, probabilities):
    plt.figure(figsize=(9, 4))
    plt.bar(labels, probabilities)
    plt.ylim(0, 1)
    plt.xlabel("Emotion")
    plt.ylabel("Probability")
    plt.title("Emotion Probability Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

# =========================================
# ALIGNMENT FUNCTIONS
# =========================================
def align_text_probabilities(raw_probs, raw_labels):
    prob_map = dict(zip([l.lower() for l in raw_labels], raw_probs))
    probs = np.array([prob_map.get(e, 0.0) for e in FINAL_EMOTIONS])
    return probs / np.sum(probs)

def align_image_probabilities(raw_probs, raw_labels):
    prob_map = {
        lbl.lower(): raw_probs[i]
        for i, lbl in enumerate(raw_labels)
    }
    probs = np.array([prob_map.get(e, 0.0) for e in FINAL_EMOTIONS])
    return probs / np.sum(probs)

def align_audio_probabilities(raw_probs):
    audio_map = dict(zip(AUDIO_ORIGINAL_LABELS, raw_probs))

    final_map = {
        "angry": audio_map.get("angry", 0.0),
        "sad": audio_map.get("sad", 0.0),
        "happy": audio_map.get("happy", 0.0),
        "neutral": audio_map.get("neutral", 0.0) + audio_map.get("calm", 0.0),
        "surprise": 0.0,
        "fear": audio_map.get("fearful", 0.0),
        "disgust": 0.0
    }

    probs = np.array([final_map[e] for e in FINAL_EMOTIONS])
    return probs / np.sum(probs)

# =========================================
# MENTAL HEALTH CALCULATION
# =========================================
def calculate_mental_health(vectors, labels):
    positive = {"happy", "neutral", "surprise"}
    negative = {"sad", "angry", "fear", "disgust"}

    pos_scores, neg_scores = [], []

    for vec in vectors:
        for i, label in enumerate(labels):
            if label in positive:
                pos_scores.append(vec[i])
            elif label in negative:
                neg_scores.append(vec[i])

    if not pos_scores:
        return "Insufficient data", 0.0

    mhi = np.mean(pos_scores) - np.mean(neg_scores)

    if mhi > 0.20:
        status = "Mentally Stable 😊"
    elif mhi > 0:
        status = "Mild Stress 😐"
    elif mhi > -0.20:
        status = "Elevated Stress 😟"
    else:
        status = "High Stress Risk ⚠️"

    return status, round(mhi, 3)

# =========================================
# COMBINED HISTORY (ORDER-INDEPENDENT)
# =========================================
def get_combined_emotion_history():
    combined = []
    for modality in ["text", "image", "audio"]:
        combined.extend(emotion_history[modality])
    return combined

# =========================================
# MAIN ROUTE
# =========================================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    plot_url = None
    mental_health = None
    mh_score = None

    if request.method == "POST":
        mode = request.form.get("mode")

        # -------- TEXT --------
        if mode == "text":
            text_input = request.form.get("text", "").strip()
            if not text_input:
                return render_template("index.html")

            outputs = text_model(
                np.array([text_input], dtype=str),
                training=False
            )

            raw_probs = outputs["classifier"].numpy()[0]
            probabilities = align_text_probabilities(raw_probs, TEXT_EMOTIONS)

        # -------- IMAGE --------
        elif mode == "image":
            file = request.files.get("file")
            if not file or file.filename == "":
                return render_template("index.html")

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            img = image.load_img(
                file_path,
                target_size=IMG_SIZE,
                color_mode="rgb"
            )

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            raw_probs = image_model.predict(img_array)[0]
            probabilities = align_image_probabilities(raw_probs, IMAGE_CLASSES)

        # -------- AUDIO --------
        elif mode == "audio":
            file = request.files.get("file")
            if not file or file.filename == "":
                return render_template("index.html")

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            with soundfile.SoundFile(file_path) as sf:
                X = sf.read(dtype="float32")
                sr = sf.samplerate
                if len(X.shape) > 1:
                    X = np.mean(X, axis=1)

            mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
            feature = mfccs.T

            feature_padded = pad_sequences(
                [feature],
                maxlen=MAX_SEQUENCE_LENGTH,
                padding="post",
                dtype="float32"
            )

            raw_probs = audio_model.predict(feature_padded)[0]
            probabilities = align_audio_probabilities(raw_probs)

        else:
            return render_template("index.html")

        # -------- FINAL OUTPUT --------
        idx = np.argmax(probabilities)
        prediction = FINAL_EMOTIONS[idx]
        confidence = f"{probabilities[idx] * 100:.2f}%"

        # store modality history
        emotion_history[mode].append(probabilities.tolist())
        if len(emotion_history[mode]) > MAX_HISTORY:
            emotion_history[mode].pop(0)

        # 🔥 COMBINED MENTAL STATE (ORDER-INDEPENDENT)
        combined_history = get_combined_emotion_history()
        if len(combined_history) >= 2:
            mental_health, mh_score = calculate_mental_health(
                combined_history,
                FINAL_EMOTIONS
            )

        plot_probabilities(FINAL_EMOTIONS, probabilities)
        plot_url = "/static/plots/emotion_plot.png"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        plot=plot_url,
        mental_health=mental_health,
        mh_score=mh_score
    )

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    app.run(debug=True)
