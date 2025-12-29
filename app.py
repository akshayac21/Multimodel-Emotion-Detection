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
# PATHS & FOLDERS
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PLOT_FOLDER = os.path.join(BASE_DIR, "static", "plots")
PLOT_PATH = os.path.join(PLOT_FOLDER, "emotion_plot.png")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# =========================================
# TEMP EMOTION MEMORY (PER MODALITY)
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

OBSERVED_EMOTIONS = [
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
    plt.figure(figsize=(8, 4))
    plt.bar(labels, probabilities)
    plt.ylim(0, 1)
    plt.xlabel("Emotion")
    plt.ylabel("Probability")
    plt.title("Emotion Probability Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

# =========================================
# MENTAL HEALTH CALCULATION
# =========================================
def calculate_mental_health(vectors, labels):
    positive = {"happy", "calm", "neutral", "surprise"}
    negative = {"sad", "angry", "fear", "fearful", "disgust"}

    pos_scores, neg_scores = [], []

    for vec in vectors:
        for i, label in enumerate(labels):
            lbl = label.lower()
            if lbl in positive:
                pos_scores.append(vec[i])
            elif lbl in negative:
                neg_scores.append(vec[i])

    if not pos_scores:
        return "Insufficient data", 0.0

    mhi = np.mean(pos_scores) - np.mean(neg_scores)

    if mhi > 0.25:
        status = "Mentally Stable 😊"
    elif mhi > 0:
        status = "Mild Stress 😐"
    elif mhi > -0.25:
        status = "Elevated Stress 😟"
    else:
        status = "High Stress Risk ⚠️"

    return status, round(mhi, 3)

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

            probabilities = outputs["classifier"].numpy()[0]
            labels = TEXT_EMOTIONS

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

            probabilities = image_model.predict(img_array)[0]
            labels = IMAGE_CLASSES

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

            probabilities = audio_model.predict(feature_padded)[0]
            labels = OBSERVED_EMOTIONS

        else:
            return render_template("index.html")

        # -------- FINAL OUTPUT --------
        idx = np.argmax(probabilities)
        prediction = labels[idx]
        confidence = f"{probabilities[idx] * 100:.2f}%"

        # store per modality (FIX)
        emotion_history[mode].append(probabilities.tolist())
        if len(emotion_history[mode]) > MAX_HISTORY:
            emotion_history[mode].pop(0)

        if len(emotion_history[mode]) >= 2:
            mental_health, mh_score = calculate_mental_health(
                emotion_history[mode],
                labels
            )

        plot_probabilities(labels, probabilities)
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
