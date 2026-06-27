<div align="center">

# 🩺 Medical AI Stethoscope

**Cardiac & Pulmonary Sound Diagnosis with CNN + GRU Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Hania_Ghouse-10B981?style=flat-square)](https://github.com/HaniaGhouse0407)

</div>

---

## 🧠 Overview

An AI-powered digital stethoscope that classifies cardiac and pulmonary conditions from auscultation audio using a CNN+GRU architecture trained on PhysioNet and PASCAL datasets. 94.2% accuracy on cardiac sounds, 98.1% on pulmonary. Built on top of the DSAi research project.

---

## ✨ Features

- ✅ CNN + GRU model — trained on 3,500+ clinical audio samples
- ✅ Cardiac: Normal, Aortic Stenosis, MVP, Mitral Regurgitation
- ✅ Pulmonary: Normal, Pneumonia, COPD, Asthma, Bronchitis
- ✅ MFCC, Mel Spectrogram, Chroma, Spectral Centroid feature extraction
- ✅ Probability breakdown per class with confidence scores
- ✅ Clean tabbed Streamlit UI with medical-grade disclaimers

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/HaniaGhouse0407/Medical-AI-Stethoscope.git
cd Medical-AI-Stethoscope

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (if needed)
cp .env.example .env
# Edit .env with your API keys

# 4. Run
streamlit run app.py
```

---

## 🛠️ Tech Stack

![streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=flat-square)  ![tensorflow](https://img.shields.io/badge/tensorflow-555555?style=flat-square)  ![librosa](https://img.shields.io/badge/librosa-555555?style=flat-square)  ![numpy](https://img.shields.io/badge/numpy-013243?style=flat-square)  ![scipy](https://img.shields.io/badge/scipy-555555?style=flat-square)  ![soundfile](https://img.shields.io/badge/soundfile-555555?style=flat-square)  ![Pillow](https://img.shields.io/badge/Pillow-555555?style=flat-square)

---

## 📁 Project Structure

```
Medical-AI-Stethoscope/
├── app.py              # Main Streamlit/Gradio application
├── requirements.txt    # Dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## 🎯 Target Roles

> Healthcare AI · Research Engineer · ML Engineer

---

<div align="center">

Made by [Hania Ghouse](https://github.com/HaniaGhouse0407) · 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/hania-ghouse/)
[![Google Scholar](https://img.shields.io/badge/Scholar-Research-4285F4?style=flat-square&logo=google-scholar)](https://scholar.google.com/citations?user=iVWuM4wAAAAJ)

</div>
