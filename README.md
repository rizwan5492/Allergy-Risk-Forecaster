# 🌿 Allergy Risk Forecaster

A Streamlit-based web application that predicts allergy risk for urban dwellers using environmental conditions (fetched in real time) and personal health factors. It leverages machine learning to provide actionable insights to individuals suffering from allergic reactions.

---

## 🚀 Features

- 🌦 Real-time weather data integration (via OpenWeatherMap API)
- 📊 Predictive allergy risk model trained on synthetic/real datasets
- 🔧 Adjustable input parameters (humidity, wind speed, pollen level, etc.)
- 🧠 Machine learning-powered predictions
- 📉 Visualization of environmental indicators
- 🔐 Secure handling of API keys using `.env` and `python-dotenv`

---

## 📦 Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Allergy-Risk-Forecaster.git
cd Allergy-Risk-Forecaster



📊 Requirements
Make sure your requirements.txt includes:

ini
Copy
Edit
streamlit==1.28.2
pandas==2.2.3
joblib==1.3.2
requests==2.31.0
numpy==1.26.4
scikit-learn==1.4.2
python-dotenv==1.0.1
matplotlib==3.8.3
seaborn==0.13.2
