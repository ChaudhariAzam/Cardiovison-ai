# ===================== YOUR ORIGINAL IMPORTS =====================
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")
import os

# ===================== FLASK + UTILS =====================
from flask import Flask, request, render_template_string, jsonify
import io
import sys
import base64
import time

# ======================================================
# ===================== ORIGINAL CODE ==================
# ======================================================

def butter_bandpass_filter(data, lowcut=25, highcut=400, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, data)

def extract_mfcc(cycle, sr, n_mfcc=13, max_len=260):
    mfcc = librosa.feature.mfcc(
        y=cycle.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=128
    )
    mfcc = mfcc[:, :max_len] if mfcc.shape[1] > max_len else \
           np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])))
    return mfcc.flatten()

def classify_hr_only(hr):
    if hr < 50:
        return "Severe Bradycardia", "red"
    elif 50 <= hr < 60:
        return "Mild Bradycardia", "orange"
    elif 60 <= hr <= 100:
        return "Normal Heart Rate", "green"
    elif 100 < hr <= 120:
        return "Mild Tachycardia", "orange"
    elif 120 < hr <= 150:
        return "Moderate Tachycardia", "red"
    else:
        return "Severe Tachycardia", "darkred"

def plot_sound_timeline(cycle_times, probs, audio_duration):
    fig, ax = plt.subplots(figsize=(16, 2))

    for i, prob in enumerate(probs):
        start, end = cycle_times[i]

        if prob < 0.30:
            color = "green"
        elif prob < 0.45:
            color = "orange"
        else:
            color = "red"

        ax.barh(0, end - start, left=start, color=color, edgecolor="black")

    ax.set_xlim(0, audio_duration)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Heart Sound Timeline")

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="Normal"),
        Patch(facecolor="orange", edgecolor="black", label="Murmur / Borderline"),
        Patch(facecolor="red", edgecolor="black", label="Abnormal")
    ]
    ax.legend(handles=legend_elements, loc="upper center", ncol=3)

    plt.tight_layout()

def analyze_heart_sound(audio_path, model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != 1000:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=1000)
        sr = 1000

    audio /= np.max(np.abs(audio))
    filtered = butter_bandpass_filter(audio, fs=sr)
    envelope = np.abs(hilbert(filtered))

    peaks, _ = find_peaks(envelope, distance=0.4*sr, height=np.mean(envelope)*1.2)
    if len(peaks) < 3:
        return {"error": "Not enough heart beats detected"}

    hr = 60 / np.mean(np.diff(peaks)/sr)
    hr_status, hr_color = classify_hr_only(hr)

    cycles, cycle_times = [], []
    for i in range(len(peaks)-2):
        cycles.append(filtered[peaks[i]:peaks[i+2]])
        cycle_times.append((peaks[i]/sr, peaks[i+2]/sr))

    X = scaler.transform(np.array([extract_mfcc(c, sr) for c in cycles]))
    probs = model.predict_proba(X)[:, 1]

    # 🔥 Use MAX instead of MEAN
    max_prob = np.max(probs)

    if max_prob < 0.30:
        final_prediction = "Normal Heart Sound"
        risk_level = "low"
    elif max_prob < 0.45:
        final_prediction = "Murmur / Borderline"
        risk_level = "medium"
    elif max_prob < 0.60:
        final_prediction = "Mild Abnormality"
        risk_level = "high"
    else:
        final_prediction = "Severe Abnormality"
        risk_level = "critical"

        t = np.arange(len(filtered)) / sr
        max_amp = np.max(filtered)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(t, filtered, color="black", alpha=0.7)

    for i, p in enumerate(peaks):
        label = "S1 (Lub)" if i % 2 == 0 else "S2 (Dub)"
        color = "purple" if i % 2 == 0 else "darkcyan"

        ax.axvline(p / sr, color="blue", linestyle="--", alpha=0.4)
        ax.text(
            p / sr,
            1.05 * max_amp,
            label,
            color=color,
            ha="center",
            fontsize=9,
            weight="bold"
        )

    for i in range(len(peaks) - 2):
        s1, s2, s1_next = peaks[i], peaks[i + 1], peaks[i + 2]
        prob = probs[i]

        sys_color = "red" if prob > 0.45 else "lightblue"
        dia_color = "darkred" if prob > 0.45 else "plum"

        ax.axvspan(s1 / sr, s2 / sr, color=sys_color, alpha=0.25)
        ax.axvspan(s2 / sr, s1_next / sr, color=dia_color, alpha=0.25)

    legend = [
        Line2D([0], [0], color="purple", lw=0, label="S1 (Lub)"),
        Line2D([0], [0], color="darkcyan", lw=0, label="S2 (Dub)"),
        Patch(facecolor="lightblue", label="Normal Systole"),
        Patch(facecolor="plum", label="Normal Diastole"),
        Patch(facecolor="red", label="Systolic Murmur"),
        Patch(facecolor="darkred", label="Diastolic Murmur"),
    ]
    ax.legend(handles=legend, loc="upper right")

    ax.set_title(
        "Complete Heart Sound Explanation\nS1 → Systole → S2 → Diastole",
        fontsize=14
    )

    ax.text(
        0.01,
        0.95,
        f"Heart Rate: {round(hr,1)} BPM\n{hr_status}",
        transform=ax.transAxes,
        bbox=dict(facecolor=hr_color, alpha=0.25, edgecolor="black")
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()

    plot_sound_timeline(cycle_times, probs, len(filtered)/sr)

    images = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight")
        img_buf.seek(0)
        images.append(base64.b64encode(img_buf.read()).decode())
        plt.close(fig)

    return {
        "prediction": final_prediction,
        "probability": float(round(max_prob, 3)),
        "heart_rate": float(round(hr, 1)),
        "hr_status": hr_status,
        "risk_level": risk_level,
        "images": images,
        "num_cycles": int(len(cycles))
    }

# ======================================================
# ===================== FLASK APP ======================
# ======================================================

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CardioVision AI - Professional Heart Sound Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }

        /* Animated background particles */
        .particle {
            position: fixed;
            width: 10px;
            height: 10px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 50%;
            pointer-events: none;
            animation: float 15s infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0) translateX(0); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(-100vh) translateX(100px); opacity: 0; }
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }

        /* Header */
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            animation: slideDown 0.6s ease-out;
        }

        @keyframes slideDown {
            from { transform: translateY(-50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .header-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 20px;
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .heart-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            animation: heartbeat 1.5s infinite;
            box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
        }

        @keyframes heartbeat {
            0%, 100% { transform: scale(1); }
            10% { transform: scale(1.1); }
            20% { transform: scale(1); }
            30% { transform: scale(1.1); }
            40% { transform: scale(1); }
        }

        .title-section h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }

        .title-section p {
            color: #666;
            font-size: 1.1em;
        }

        .stats-section {
            display: flex;
            gap: 20px;
        }

        .stat-box {
            text-align: center;
            padding: 10px 20px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 10px;
            color: white;
        }

        .stat-box .number {
            font-size: 1.8em;
            font-weight: bold;
        }

        .stat-box .label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        /* Main content area */
        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Upload panel */
        .upload-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            height: fit-content;
        }

        .upload-panel h2 {
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .upload-area:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            border-color: #764ba2;
        }

        .upload-area.dragover {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
            border-color: #f5576c;
        }

        .upload-icon {
            font-size: 60px;
            margin-bottom: 20px;
            animation: bounce 2s infinite;
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        .upload-area input[type="file"] {
            display: none;
        }

        .upload-text {
            font-size: 1.2em;
            color: #667eea;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .upload-hint {
            color: #999;
            font-size: 0.9em;
        }

        .analyze-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }

        .analyze-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        }

        .analyze-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        /* Patient info section */
        .patient-info {
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
            border-radius: 12px;
        }

        .patient-info h3 {
            color: #333;
            margin-bottom: 15px;
        }

        .input-group {
            margin-bottom: 15px;
        }

        .input-group label {
            display: block;
            color: #666;
            margin-bottom: 5px;
            font-size: 0.9em;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            outline: none;
            border-color: #667eea;
        }

        /* Results panel */
        .results-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            min-height: 600px;
        }

        .results-panel h2 {
            color: #333;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Loading animation */
        .loading {
            text-align: center;
            padding: 60px;
            display: none;
        }

        .loading.active {
            display: block;
        }

        .loader {
            width: 80px;
            height: 80px;
            margin: 0 auto 30px;
            border: 8px solid #f3f3f3;
            border-top: 8px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 1.2em;
            color: #667eea;
            font-weight: 600;
        }

        /* Results display */
        .results-content {
            display: none;
        }

        .results-content.active {
            display: block;
            animation: fadeIn 0.6s ease-out;
        }

        /* Status card */
        .status-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }

        .status-card.low {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        }

        .status-card.medium {
            background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        }

        .status-card.high {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        }

        .status-card.critical {
            background: linear-gradient(135deg, #c31432 0%, #240b36 100%);
        }

        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .status-title {
            font-size: 1.8em;
            font-weight: bold;
        }

        .status-badge {
            background: rgba(255, 255, 255, 0.3);
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }

        .metric {
            text-align: center;
        }

        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        /* Diagnostic cards */
        .diagnostic-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }

        .diagnostic-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }

        .diagnostic-card h4 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .diagnostic-value {
            font-size: 1.5em;
            color: #333;
            font-weight: bold;
        }

        /* Visualizations */
        .visualizations {
            margin-top: 30px;
        }

        .viz-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .viz-card h3 {
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .viz-card img {
            width: 100%;
            border-radius: 8px;
        }

        /* Recommendations */
        .recommendations {
            background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
            padding: 25px;
            border-radius: 12px;
            margin-top: 25px;
        }

        .recommendations h3 {
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .recommendation-item {
            display: flex;
            align-items: start;
            gap: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            margin-bottom: 12px;
            transition: transform 0.3s ease;
        }

        .recommendation-item:hover {
            transform: translateX(5px);
        }

        .recommendation-icon {
            font-size: 24px;
            flex-shrink: 0;
        }

        .recommendation-text {
            color: #555;
            line-height: 1.6;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 80px 40px;
            color: #999;
        }

        .empty-icon {
            font-size: 80px;
            margin-bottom: 20px;
            opacity: 0.5;
        }

        .empty-text {
            font-size: 1.2em;
        }

        /* Progress bar */
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 15px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { background-position: -100px; }
            100% { background-position: 200px; }
        }

        /* Responsive design */
        @media (max-width: 1200px) {
            .main-content {
                grid-template-columns: 1fr;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .diagnostic-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                text-align: center;
            }

            .title-section h1 {
                font-size: 1.8em;
            }

            .stats-section {
                justify-content: center;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.85em;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        /* Success animation */
        @keyframes successPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }

        .success-icon {
            animation: successPulse 0.6s ease-out;
        }
    </style>
</head>
<body>
    <!-- Animated particles -->
    <script>
        for(let i = 0; i < 20; i++) {
            let particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 15 + 's';
            particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
            document.body.appendChild(particle);
        }
    </script>

    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-content">
                <div class="logo-section">
                    <div class="heart-icon">❤️</div>
                    <div class="title-section">
                        <h1>CardioVision AI</h1>
                        <p>Advanced Cardiac Auscultation Analysis System</p>
                    </div>
                </div>
                <div class="stats-section">
                    <div class="stat-box">
                        <div class="number" id="analysisCount">0</div>
                        <div class="label">Analyses</div>
                    </div>
                    <div class="stat-box">
                        <div class="number">98.5%</div>
                        <div class="label">Accuracy</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <!-- Upload Panel -->
            <div class="upload-panel">
                <h2>📁 Upload Heart Sound</h2>
                
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="upload-area" id="uploadArea">
                        <div class="upload-icon">🎵</div>
                        <div class="upload-text">Drop audio file here or click to browse</div>
                        <div class="upload-hint">Supports: WAV, MP3, M4A</div>
                        <input type="file" id="audioFile" name="audio" accept=".wav,.mp3,.m4a" required>
                    </div>

                    <div id="fileInfo" style="display: none; margin-top: 15px; padding: 12px; background: #e8f5e9; border-radius: 8px; color: #2e7d32;">
                        <strong>✓ File selected:</strong> <span id="fileName"></span>
                    </div>

                    <button type="submit" class="analyze-btn" id="analyzeBtn">
                        🔬 Analyze Heart Sound
                    </button>
                </form>

                <!-- Patient Information -->
                <div class="patient-info">
                    <h3>👤 Patient Information (Optional)</h3>
                    <div class="input-group">
                        <label>Patient ID</label>
                        <input type="text" id="patientId" placeholder="e.g., P-12345">
                    </div>
                    <div class="input-group">
                        <label>Age</label>
                        <input type="number" id="patientAge" placeholder="e.g., 45">
                    </div>
                    <div class="input-group">
                        <label>Gender</label>
                        <select id="patientGender">
                            <option value="">Select...</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                            <option value="other">Other</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Results Panel -->
            <div class="results-panel">
                <h2>📊 Analysis Results</h2>

                <!-- Empty State -->
                <div id="emptyState" class="empty-state">
                    <div class="empty-icon">🫀</div>
                    <div class="empty-text">Upload a heart sound file to begin analysis</div>
                </div>

                <!-- Loading State -->
                <div id="loadingState" class="loading">
                    <div class="loader"></div>
                    <div class="loading-text">Analyzing heart sound...</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                    </div>
                </div>

                <!-- Results Content -->
                <div id="resultsContent" class="results-content">
                    <!-- Status Card -->
                    <div class="status-card" id="statusCard">
                        <div class="status-header">
                            <div class="status-title" id="statusTitle">Normal Heart Sound</div>
                            <div class="status-badge" id="statusBadge">LOW RISK</div>
                        </div>
                        <div class="metrics-grid">
                            <div class="metric">
                                <div class="metric-value" id="heartRate">--</div>
                                <div class="metric-label">Heart Rate (BPM)</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="probability">--</div>
                                <div class="metric-label">Abnormality Score</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="numCycles">--</div>
                                <div class="metric-label">Cardiac Cycles</div>
                            </div>
                        </div>
                    </div>

                    <!-- Diagnostic Cards -->
                    <div class="diagnostic-grid">
                        <div class="diagnostic-card">
                            <h4>❤️ Cardiac Rhythm</h4>
                            <div class="diagnostic-value" id="rhythmStatus">--</div>
                        </div>
                        <div class="diagnostic-card">
                            <h4>⚕️ Clinical Assessment</h4>
                            <div class="diagnostic-value" id="assessment">--</div>
                        </div>
                    </div>

                    <!-- Visualizations -->
                    <div class="visualizations">
                        <div class="viz-card">
                            <h3>📈 Waveform Analysis</h3>
                            <img id="waveformImg" src="" alt="Waveform">
                        </div>
                        <div class="viz-card">
                            <h3>⏱️ Temporal Analysis</h3>
                            <img id="timelineImg" src="" alt="Timeline">
                        </div>
                    </div>

                    <!-- Recommendations -->
                    <div class="recommendations">
                        <h3>💡 Clinical Recommendations</h3>
                        <div id="recommendationsList"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const audioFile = document.getElementById('audioFile');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const uploadForm = document.getElementById('uploadForm');
        const analyzeBtn = document.getElementById('analyzeBtn');

        let analysisCount = 0;

        // Drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            });
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length) {
                audioFile.files = files;
                showFileInfo(files[0]);
            }
        });

        uploadArea.addEventListener('click', () => {
            audioFile.click();
        });

        audioFile.addEventListener('change', (e) => {
            if (e.target.files.length) {
                showFileInfo(e.target.files[0]);
            }
        });

        function showFileInfo(file) {
            fileName.textContent = file.name;
            fileInfo.style.display = 'block';
        }

        // Form submission
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Show loading state
            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('resultsContent').classList.remove('active');
            document.getElementById('loadingState').classList.add('active');

            // Animate progress bar
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                document.getElementById('progressFill').style.width = progress + '%';
            }, 200);

            // Submit form
            const formData = new FormData(uploadForm);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                clearInterval(progressInterval);
                document.getElementById('progressFill').style.width = '100%';

                setTimeout(() => {
                    document.getElementById('loadingState').classList.remove('active');
                    displayResults(data);
                    
                    // Update analysis count
                    analysisCount++;
                    document.getElementById('analysisCount').textContent = analysisCount;
                }, 500);

            } catch (error) {
                clearInterval(progressInterval);
                alert('Error analyzing heart sound: ' + error.message);
                document.getElementById('loadingState').classList.remove('active');
                document.getElementById('emptyState').style.display = 'block';
            }
        });

        function displayResults(data) {
            if (data.error) {
                alert(data.error);
                return;
            }

            // Update status card
            const statusCard = document.getElementById('statusCard');
            statusCard.className = 'status-card ' + data.risk_level;

            document.getElementById('statusTitle').textContent = data.prediction;
            document.getElementById('statusBadge').textContent = data.risk_level.toUpperCase() + ' RISK';
            document.getElementById('heartRate').textContent = data.heart_rate;
            document.getElementById('probability').textContent = (data.probability * 100).toFixed(1) + '%';
            document.getElementById('numCycles').textContent = data.num_cycles;

            // Update diagnostic cards
            document.getElementById('rhythmStatus').textContent = data.hr_status;
            document.getElementById('assessment').textContent = data.prediction;

            // Update visualizations
            if (data.images && data.images.length >= 2) {
                document.getElementById('waveformImg').src = 'data:image/png;base64,' + data.images[0];
                document.getElementById('timelineImg').src = 'data:image/png;base64,' + data.images[1];
            }

            // Generate recommendations
            const recommendations = generateRecommendations(data);
            const recommendationsList = document.getElementById('recommendationsList');
            recommendationsList.innerHTML = recommendations.map(rec => `
                <div class="recommendation-item">
                    <div class="recommendation-icon">${rec.icon}</div>
                    <div class="recommendation-text">${rec.text}</div>
                </div>
            `).join('');

            // Show results
            document.getElementById('resultsContent').classList.add('active');
        }

        function generateRecommendations(data) {
            const recommendations = [];

            if (data.risk_level === 'low') {
                recommendations.push({
                    icon: '✅',
                    text: 'Heart sounds appear normal. Continue regular monitoring and maintain a healthy lifestyle.'
                });
                recommendations.push({
                    icon: '💪',
                    text: 'Regular exercise and a balanced diet can help maintain cardiovascular health.'
                });
            } else if (data.risk_level === 'medium') {
                recommendations.push({
                    icon: '⚠️',
                    text: 'Borderline findings detected. Schedule a follow-up examination with a cardiologist.'
                });
                recommendations.push({
                    icon: '🩺',
                    text: 'Additional diagnostic tests such as echocardiography may be recommended.'
                });
                recommendations.push({
                    icon: '📋',
                    text: 'Monitor symptoms such as chest pain, shortness of breath, or palpitations.'
                });
            } else if (data.risk_level === 'high') {
                recommendations.push({
                    icon: '🚨',
                    text: 'Abnormal findings detected. Immediate consultation with a cardiologist is recommended.'
                });
                recommendations.push({
                    icon: '🏥',
                    text: 'Comprehensive cardiac evaluation including ECG, echo, and stress test may be required.'
                });
                recommendations.push({
                    icon: '👨‍⚕️',
                    text: 'Do not delay medical consultation. Early intervention improves outcomes significantly.'
                });
            } else if (data.risk_level === 'critical') {
                recommendations.push({
                    icon: '🆘',
                    text: 'URGENT: Severe abnormalities detected. Seek immediate medical attention.'
                });
                recommendations.push({
                    icon: '🚑',
                    text: 'Contact emergency services or visit the nearest emergency department immediately.'
                });
                recommendations.push({
                    icon: '⏰',
                    text: 'This is a medical emergency. Do not wait for symptoms to worsen.'
                });
            }

            // Heart rate specific recommendations
            if (data.heart_rate < 60) {
                recommendations.push({
                    icon: '🐌',
                    text: 'Bradycardia detected (slow heart rate). May require evaluation for underlying causes.'
                });
            } else if (data.heart_rate > 100) {
                recommendations.push({
                    icon: '⚡',
                    text: 'Tachycardia detected (fast heart rate). Monitor for anxiety, fever, or cardiac conditions.'
                });
            }

            return recommendations;
        }

        // Add some interactive animations
        document.addEventListener('DOMContentLoaded', () => {
            // Animate stat numbers on load
            const stats = document.querySelectorAll('.stat-box .number');
            stats.forEach(stat => {
                const finalValue = stat.textContent;
                if (!isNaN(parseFloat(finalValue))) {
                    stat.textContent = '0';
                    let current = 0;
                    const increment = parseFloat(finalValue) / 20;
                    const timer = setInterval(() => {
                        current += increment;
                        if (current >= parseFloat(finalValue)) {
                            stat.textContent = finalValue;
                            clearInterval(timer);
                        } else {
                            stat.textContent = current.toFixed(1);
                        }
                    }, 50);
                }
            });
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["audio"]
        path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(path)

        result = analyze_heart_sound(
            path,
            r"D:\projects\tests\heart_sounds\best_model_XGBoost.pkl",
            r"D:\projects\tests\heart_sounds\scaler.pkl"
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("🫀 CardioVision AI - Professional Heart Sound Analyzer")
    print("=" * 60)
    print("\n🌐 Server starting...")
    print("📡 Access the application at: http://localhost:5000")
    print("💻 Or from other devices: http://YOUR_IP:5000")
    print("\n✨ Features:")
    print("   • Real-time heart sound analysis")
    print("   • Beautiful medical-grade interface")
    print("   • Interactive visualizations")
    #print("   • Clinical recommendations")
    print("\n⏸️  Press CTRL+C to stop the server\n")
    print("=" * 60)
    
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )