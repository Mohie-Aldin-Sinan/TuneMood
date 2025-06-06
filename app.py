import gradio as gr
import pandas as pd
import numpy as np
import librosa
import joblib
import tempfile
import os
import scipy.stats

# Load model and scaler
model = joblib.load("TuneMood_model.pkl")
scaler = joblib.load("TuneMood_scaler.pkl")

# Load dataset
url = "https://raw.githubusercontent.com/Mohie-Aldin-Sinan/TuneMood/main/data/Acoustic%20Features.csv"
df = pd.read_csv(url)

# Feature columns and label
feature_columns = df.select_dtypes(include=np.number).columns.tolist()
label_column = df.columns.difference(feature_columns)[0]

# Extract Features Function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    rms = np.mean(librosa.feature.rms(y=y))
    rms_frames = librosa.feature.rms(y=y)[0]
    low_energy = np.mean(rms_frames < np.median(rms_frames) * 0.5)
    fluctuation = np.std(rms_frames)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    roughness_mean = np.mean(onset_env)
    times = np.arange(len(onset_env))
    roughness_slope = np.polyfit(times, onset_env, 1)[0] if len(times) > 1 else 0.0
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    attack_time = np.argmax(onset_env) / sr
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    duration = librosa.get_duration(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    event_density = len(onset_frames) / duration if duration > 0 else 0
    ac = librosa.autocorrelate(y)
    pulse_clarity = np.max(ac[1:]) / ac[0] if ac[0] != 0 else 0
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    brightness = np.mean(spec_centroid) / (sr / 2)
    spectral_centroid = np.mean(spec_centroid)
    spectral_spread = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    S = np.abs(librosa.stft(y))
    spectral_skewness = scipy.stats.skew(S, axis=0).mean()
    spectral_kurtosis = scipy.stats.kurtosis(S, axis=0).mean()
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    psd = S**2
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-8)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8), axis=0)
    entropy_mean = np.mean(entropy)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = np.mean(chroma, axis=1)
    y_harm, y_perc = librosa.effects.hpss(y)
    hcd_mean = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-8)
    hcd_std = np.std(np.abs(y_harm))
    harmonic_env = librosa.onset.onset_strength(y=y_harm, sr=sr)
    hcd_slope = np.polyfit(np.arange(len(harmonic_env)), harmonic_env, 1)[0] if len(harmonic_env) > 1 else 0.0
    ac_full = librosa.autocorrelate(y)
    peak_idx = np.argmax(ac_full[1:]) + 1
    period_freq = np.log1p(sr / peak_idx) if peak_idx != 0 else 0
    period_amp = np.log1p(ac_full[peak_idx])
    window = ac_full[max(0, peak_idx - 5):peak_idx + 6]
    p = window / (np.sum(window) + 1e-8)
    period_entropy = -np.sum(p * np.log2(p + 1e-8))

    return np.hstack([
        rms, low_energy, fluctuation, tempo, mfcc_means,
        roughness_mean, roughness_slope, zcr, attack_time, roughness_slope,
        rolloff, event_density, pulse_clarity, brightness, spectral_centroid,
        spectral_spread, spectral_skewness, spectral_kurtosis, spectral_flatness,
        entropy_mean, chroma_means, hcd_mean, hcd_std, hcd_slope,
        period_freq, period_amp, period_entropy
    ])

# Prediction function
def predict_mood(input_type, sample_index, audio_file):
    try:
        if input_type == "From Dataset":
            idx = int(sample_index)
            if idx < 0 or idx >= len(df):
                return f"❌ Sample index out of range. Please enter between 0 and {len(df)-1}."
            row = df.iloc[idx]
            features = row[feature_columns].values.reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            return f"🎵 Predicted Mood for Sample #{idx}: **{prediction}** (Actual: {row[label_column]})"
        elif input_type == "From Audio File":
            if not audio_file:
                return "❌ Please upload a .mp3 or .wav file."

            features = extract_features(audio_file)
            if np.isnan(features).any():
                features = np.nan_to_num(features)
            features_scaled = scaler.transform(pd.DataFrame([features], columns=feature_columns))
            prediction = model.predict(features_scaled)[0]
            return f"🎧 Predicted Mood: **{prediction}**"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Show/hide fields
def update_inputs(choice):
    return (
        gr.update(visible=(choice == "From Dataset")),
        gr.update(visible=(choice == "From Audio File")),
    )

# UI
with gr.Blocks(css="""
#predict_button {
    background-color: #007BFF;
    color: white;
    font-weight: bold;
}
#predict_button:hover {
    background-color: #004aad;
}
""") as demo:
    gr.Markdown("## 🎵 TuneMood - Mood Prediction from Music")
    gr.Markdown('[Download/Reference Dataset](https://github.com/Mohie-Aldin-Sinan/TuneMood/blob/main/data/Acoustic%20Features.csv)')

    input_choice = gr.Radio(["From Dataset", "From Audio File"], label="Select Input Type", value="From Dataset")

    with gr.Row():
        sample_index = gr.Number(label="Enter Sample Index", visible=True, value=0, precision=0)
        audio_upload = gr.Audio(type="filepath", label="Upload MP3 or WAV File", visible=False)

    submit_btn = gr.Button("Predict Mood", elem_id="predict_button")
    output_text = gr.Markdown()

    input_choice.change(fn=update_inputs, inputs=input_choice, outputs=[sample_index, audio_upload])

    submit_btn.click(fn=lambda: "⏳ Predicting mood, please wait...", inputs=[], outputs=output_text, queue=False)
    submit_btn.click(fn=predict_mood, inputs=[input_choice, sample_index, audio_upload], outputs=output_text)

demo.launch(share=True)
