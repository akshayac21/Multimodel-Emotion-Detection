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
# GEMINI IMPORT (FIXED)
# =========================================
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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
# GEMINI CONFIG
# =========================================
os.environ["GEMINI_API_KEY"] = "Provide your gemini api here"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

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
    prob_map = {lbl.lower(): raw_probs[i] for i, lbl in enumerate(raw_labels)}
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
# GEMINI SUGGESTIONS (FIXED SAFETY)
# =========================================
def generate_ai_suggestions(mental_state, mh_score):
    prompt = f"""
Act as a supportive wellness coach.(just for educational purpose)
The user's mental state is: {mental_state} (Index Score: {mh_score}).

Provide 3-5 short, actionable, non-clinical wellness tips.
Plain text only.
One tip per line.
No bullet points.
No medical diagnosis.
"""

    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        response = gemini_model.generate_content(
            prompt,
            safety_settings=safety_settings
        )

        if response.text:
            return response.text.strip()

    except Exception as e:
        print("Gemini API Error:", e)

    return (
        "Take a few deep breaths\n"
        "Stay hydrated\n"
        "Step away from screens briefly\n"
        "Reach out to someone you trust\n"
        "(Api error occured)"
    )

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
    ai_suggestions = None

    if request.method == "POST":
        mode = request.form.get("mode")

        if mode == "text":
            text_input = request.form.get("text", "").strip()
            outputs = text_model(np.array([text_input], dtype=str), training=False)
            probabilities = align_text_probabilities(
                outputs["classifier"].numpy()[0], TEXT_EMOTIONS
            )

        elif mode == "image":
            file = request.files.get("file")
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=IMG_SIZE)
            img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
            probabilities = align_image_probabilities(
                image_model.predict(img_array)[0], IMAGE_CLASSES
            )

        elif mode == "audio":
            file = request.files.get("file")
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            with soundfile.SoundFile(file_path) as sf:
                X = sf.read(dtype="float32")
                sr = sf.samplerate
                if len(X.shape) > 1:
                    X = np.mean(X, axis=1)

            mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
            feature_padded = pad_sequences(
                [mfccs.T], maxlen=MAX_SEQUENCE_LENGTH, padding="post"
            )
            probabilities = align_audio_probabilities(
                audio_model.predict(feature_padded)[0]
            )

        idx = np.argmax(probabilities)
        prediction = FINAL_EMOTIONS[idx]
        confidence = f"{probabilities[idx] * 100:.2f}%"

        emotion_history[mode].append(probabilities.tolist())
        if len(emotion_history[mode]) > MAX_HISTORY:
            emotion_history[mode].pop(0)

        combined = (
            emotion_history["text"]
            + emotion_history["image"]
            + emotion_history["audio"]
        )

        if len(combined) >= 2:
            mental_health, mh_score = calculate_mental_health(combined, FINAL_EMOTIONS)
            ai_suggestions = generate_ai_suggestions(mental_health, mh_score)

        plot_probabilities(FINAL_EMOTIONS, probabilities)
        plot_url = "/static/plots/emotion_plot.png"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        plot=plot_url,
        mental_health=mental_health,
        mh_score=mh_score,
        ai_suggestions=ai_suggestions
    )

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    app.run(debug=True)
