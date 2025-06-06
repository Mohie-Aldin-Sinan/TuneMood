---
title: TuneMood 🎵
emoji: 🎧
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.27.0"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🎵 TuneMood – Predict the Mood of Music with ML

TuneMood is a machine learning web app that predicts the **mood** of a song using its acoustic features or raw audio. Built with **Gradio** and powered by a trained classification model, it allows you to:

- 🎧 Upload a `.mp3` or `.wav` file  
- 📊 Select a sample from the dataset  
- 💡 Instantly get a predicted mood like *Happy*, *Sad*, *Angry*, or *Calm*

---

## 🚀 Try It Live

👉 [**Launch TuneMood on Hugging Face Spaces**](https://huggingface.co/spaces/Mohie-Aldin-Sinan/TuneMood)

---

## 🧠 How It Works

The model was trained on a dataset of audio tracks labeled with mood categories. Features were extracted using the `librosa` audio analysis library, including:

- Root Mean Square (RMS) Energy  
- Tempo & Beat Metrics  
- Spectral Features (Centroid, Contrast, Bandwidth)  
- Zero Crossing Rate  
- Entropy of Energy and Tempo  

The model is trained using scikit-learn and scaled for consistency using a `StandardScaler`.

---

## 🧪 Example Moods Predicted

- Happy  
- Sad  
- Angry  
- Calm

---

## 📁 How to Use

1. **Choose Input Method**:
   - `From Dataset`: Enter a sample index to test from the dataset
   - `From Audio File`: Upload your own music file (`.mp3` or `.wav`)

2. **Click "Predict Mood"**:
   - The model analyzes the features and returns a mood label

---

## 📦 Tech Stack

- 🧠 **Machine Learning**: Scikit-learn
- 🎼 **Feature Extraction**: Librosa
- 🖼️ **Frontend & UI**: Gradio
- ☁️ **Deployment**: Hugging Face Spaces

---

## 📚 Dataset Reference

- [Download Dataset CSV](https://github.com/Mohie-Aldin-Sinan/TuneMood/blob/main/data/Acoustic%20Features.csv)

---

## 🛠️ Local Setup

```bash
# Clone the repo
git clone https://huggingface.co/spaces/Mohie-Aldin-Sinan/TuneMood
cd TuneMood

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
