---
title: TuneMood
emoji: 🎵
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
short_description: 'TuneMood: Predict the Mood of a Song Using AI'
---

# 🎧 TuneMood — Music Mood Prediction

TuneMood is an AI-powered web app that predicts the **mood** of a song from either its acoustic features or raw audio input.

Built with **Gradio** and a trained machine learning model, it helps you quickly identify moods like *Happy*, *Sad*, *Angry*, or *Calm* from music samples or your own audio files.

---

## 🚀 How to Use

- Choose input type:
  - **From Dataset** — enter a sample index to test on existing music samples.
  - **From Audio File** — upload your own `.mp3` or `.wav` music file.

- Click **Predict Mood** to see the model's prediction instantly.

---

## 🧠 Behind the Scenes

- Extracts features using the powerful [Librosa](https://librosa.org/) library.
- Uses features like RMS energy, tempo, spectral properties, and entropy.
- Runs predictions on a trained classification model built with scikit-learn.
- Scales features for consistency with a `StandardScaler`.

---

## ⚙️ Tech Stack

- Python, scikit-learn for ML  
- Librosa for audio feature extraction  
- Gradio for the user interface  
- Hugging Face Spaces for deployment

---

## 📂 Dataset, Model & Deployment

- Dataset with acoustic features available [here](https://github.com/Mohie-Aldin-Sinan/TuneMood/blob/main/data/Acoustic%20Features.csv)  
- Model and scaler are pre-trained and loaded internally.  
- Try the live app here: [TuneMood Live Demo](https://huggingface.co/spaces/Mohie-Aldin-Sinan/TuneMood)

---

## 📚 Learn More

Check out the full [GitHub repo](https://github.com/Mohie-Aldin-Sinan/TuneMood) for source code, dataset, and training details.

---

Made with 💜 by Mohie Aldin Sinan

---

*Feel free to try it out and share your feedback!*

