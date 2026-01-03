# Emo AI – Multimodal Emotion and Mental Health Analysis System

## Overview

**Emo AI** is a Flask-based multimodal artificial intelligence application designed to analyze human emotions and mental well-being using three independent modalities:

* Text emotion analysis
* Facial emotion recognition from images
* Speech emotion recognition from audio

The system fuses probability-based emotion outputs across modalities and over time to compute a **Mental Health Index (MHI)** and generates **non-clinical wellness suggestions** using the Google Gemini API.

This project is intended strictly for **educational and research purposes** and does not provide medical diagnosis.

---

## Key Features

* Multimodal emotion detection (Text, Image, Audio)
* Unified emotion probability distribution across all models
* Emotion history tracking per modality
* Mental Health Index based on emotional trends
* AI-generated wellness suggestions using Gemini
* Interactive professional dashboard
* CPU-compatible inference (no GPU required at runtime)

---

## Unified Emotion Classes

All models are aligned to the following seven emotion categories:

```
angry, sad, happy, neutral, surprise, fear, disgust
```

---

## Project Structure

```
project_root/
│
├── app.py                          # Main Flask application
│
├── models/                         # Pre-trained models
│   ├── audio/
│   │   └── voice_emotion_cnn_bilstm.h5
│   │
│   ├── image/
│   │   └── fer2013_emotion_model_final.h5
│   │
│   └── text/
│       ├── emotion_bert_model/
│       │   ├── assets/
│       │   │   └── vocab.txt
│       │   ├── variables/
│       │   │   ├── variables.data-00000-of-00001
│       │   │   └── variables.index
│       │   ├── fingerprint.pb
│       │   ├── keras_metadata.pb
│       │   └── saved_model.pb
│       └── label_encoder.pkl
│
├── static/
│   ├── css/style.css
│   ├── js/ui.js
│   └── plots/
│
├── templates/
│   └── index.html
│
├── training/
│   ├── train_audio.ipynb
│   ├── train_image.ipynb
│   └── train_text.ipynb
│
├── uploads/
│
├── screenshots/
│
├── requirements.txt
│
└── README.md
```

---

## Important Project Integrity Notice

**All files and folders other than the following must remain unchanged:**

* `screenshots/`
* `uploads/`
* `requirements.txt` (only for dependency updates if required)

### Strict Instructions

* Do **not** rename files or folders
* Do **not** change model filenames
* Do **not** modify directory hierarchy
* Do **not** move models outside the `models/` directory
* Do **not** alter training notebooks unless retraining is intended

The application logic in `app.py` is tightly coupled with the above structure.
Any deviation may result in runtime errors or incorrect model loading.

---

## Datasets Used for Training

All models were trained using **Kaggle Notebooks**, which provide direct dataset access and free GPU resources.

### Text Emotion Detection Model

* Model Type: BERT-based emotion classifier
* Datasets:

  * GoEmotions (Google)
  * Emotion-labeled conversational datasets
* Training Notebook: `training/train_text.ipynb`
* Output:

  * TensorFlow SavedModel
  * Label encoder (`label_encoder.pkl`)

---

### Facial Emotion Recognition Model

* Model Type: Convolutional Neural Network
* Dataset:

  * FER-2013 (Kaggle)
* Classes:

  * Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
* Training Notebook: `training/train_image.ipynb`
* Output:

  * Keras `.h5` model

---

### Speech Emotion Recognition Model

* Model Type: CNN + BiLSTM
* Datasets:

  * RAVDESS
  * SAVEE
  * TESS
* Training Notebook: `training/train_audio.ipynb`
* Output:

  * Keras `.h5` model

---

## Training Recommendation

Model training was performed in **Kaggle Notebooks** due to:

* Free GPU availability
* Built-in dataset hosting
* Stable TensorFlow execution environment

### Training Workflow

1. Upload the training notebook to Kaggle
2. Attach the respective dataset
3. Enable GPU
4. Run all cells
5. Download trained model files
6. Update the trained files in the exact project structure shown above

No retraining is required to run the application.

---

## System Requirements

| Component        | Requirement       |
| ---------------- | ----------------- |
| Operating System | Ubuntu / Linux    |
| Python Version   | 3.11 or lower     |
| GPU              | Optional          |
| Browser          | Chrome or Firefox |

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd project_root
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Gemini API Configuration

This application uses the **Google Gemini API** for generating wellness suggestions.

### Add Your API Key

Open `app.py` and locate:

```python
os.environ["GEMINI_API_KEY"] = "Provide your gemini api here"
```

Replace it with:

```python
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
```

Do not expose API keys in public repositories.

---

## Running the Application (Linux / Ubuntu)

```bash
python app.py
```

Access the application at:

```
http://127.0.0.1:5000
```

---

## Application Usage

1. Select input modality (Text, Image, Audio)
2. Provide input data
3. Click **Analyze Data**
4. View:

   * Predicted emotion
   * Confidence score
   * Probability distribution
   * Mental Health Index (after two or more analyses)
   * AI-generated wellness suggestions

---

## Screenshots Reference

The following screenshots demonstrate the complete application workflow:

* Home page interface and application overview
  `(screenshots/home.png)`

* Facial emotion analysis image upload and preview interface
  `(screenshots/image_input.png)`

* Facial emotion analysis results 
  `(screenshots/image_result.png)`

* Speech emotion analysis audio upload and preview interface
  `(screenshots/audio_input.png)`

* Speech emotion analysis results 
  `(screenshots/audio_result.png)`
  
* Text emotion analysis input interface
  `(screenshots/text_input.png)`

* Text emotion analysis results 
  `(screenshots/text_result.png)`

---

## Ethical and Usage Notice

* No medical diagnosis is provided
* AI outputs are non-clinical
* No permanent personal data storage
* Intended strictly for academic and research use

---

## Disclaimer

This application is developed solely for educational and research purposes. It is not a medical or diagnostic tool. The emotion analysis results and AI-generated suggestions are informational only and should not be interpreted as professional medical, psychological, or clinical advice. Users should seek qualified healthcare professionals for any mental health concerns.

---

## Auther
AKSHAY A C
