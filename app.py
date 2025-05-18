import streamlit as st
import pandas as pd
import joblib
import requests
import time

# Load model with error handling
try:
    model = joblib.load("allergy_risk_model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# OpenWeatherMap API key
API_KEY = "7ed4dac95a65b09c3ca0b3c144b97ac9"  # Your actual API key

st.title("Allergy Risk Forecaster for Urban Dwellers")
st.write("Enter your details and city to predict allergy risk and get a daily action plan. The app will remain open until you stop it manually.")

# Display initial instructions using Markdown
st.write("### Instructions")
st.markdown("""
- Enter your city or ZIP code and adjust the sliders/dropdowns.
- Check 'Fetch Real-Time AQI and Weather' to use live data (requires a valid API key).
- Click 'Predict Allergy Risk' to see your results.
""")
st.write("Note: The results will stay until you clear them manually.")

# Input form
with st.form("allergy_form"):
    city = st.text_input("City or ZIP Code", "New York")
    pollen_exposure = st.slider("Hours Spent Outdoors", 0, 12, 2)
    pollution_exposure = st.selectbox("Proximity to Traffic/Industry", ["Low", "Medium", "High"])
    pet_exposure = st.slider("Hours with Pets", 0, 8, 1)
    dietary_triggers = st.selectbox("Allergenic Food Consumption", ["None", "Occasional", "Frequent"])
    medication_use = st.selectbox("Antihistamine Use", ["None", "Occasional", "Regular"])
    ventilation = st.selectbox("Indoor Ventilation Quality", ["Poor", "Average", "Good"])
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    fetch_api = st.checkbox("Fetch Real-Time AQI and Weather")
    submit = st.form_submit_button("Predict Allergy Risk")

# Fetch real-time data with improved error handling
aqi = None
temperature = None
humidity = None
if fetch_api and submit:
    with st.spinner("Fetching real-time weather data..."):
        try:
            lat, lon = 40.7128, -74.0060  # Dummy coordinates for New York
            url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
            response = requests.get(url, timeout=10).json()
            aqi = response["list"][0]["main"]["aqi"] * 100  # Scale to 0–500
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
            weather = requests.get(weather_url, timeout=10).json()
            temperature = weather["main"]["temp"] - 273.15  # Kelvin to Celsius
            humidity = weather["main"]["humidity"]
            st.success(f"AQI: {aqi}, Temperature: {temperature:.1f}°C, Humidity: {humidity}%")
        except Exception as e:
            st.error(f"API Error: {e}. Using manual inputs instead.")
            aqi = st.slider("Enter AQI Manually (0–500)", 0, 500, 100)
            temperature = st.slider("Enter Temperature (°C)", -10, 40, 25)
            humidity = st.slider("Enter Humidity (%)", 0, 100, 60)
else:
    aqi = st.slider("Enter AQI Manually (0–500)", 0, 500, 100)
    temperature = st.slider("Enter Temperature (°C)", -10, 40, 25)
    humidity = st.slider("Enter Humidity (%)", 0, 100, 60)

# Initialize session state for results
if "results_shown" not in st.session_state:
    st.session_state.results_shown = False
    st.session_state.risk_score = None
    st.session_state.tips = []
    st.session_state.chart_data = None

# Predict and store results
if submit:
    with st.spinner("Calculating allergy risk..."):
        time.sleep(2)  # Simulate processing time for better UX
        input_data = pd.DataFrame({
            "pollen_exposure": [pollen_exposure],
            "aqi": [aqi],
            "pollution_Low": [1 if pollution_exposure == "Low" else 0],
            "pollution_Medium": [1 if pollution_exposure == "Medium" else 0],
            "pollution_High": [1 if pollution_exposure == "High" else 0],
            "pet_exposure": [pet_exposure],
            "dietary_None": [1 if dietary_triggers == "None" else 0],
            "dietary_Occasional": [1 if dietary_triggers == "Occasional" else 0],
            "dietary_Frequent": [1 if dietary_triggers == "Frequent" else 0],
            "medication_None": [1 if medication_use == "None" else 0],
            "medication_Occasional": [1 if medication_use == "Occasional" else 0],
            "medication_Regular": [1 if medication_use == "Regular" else 0],
            "ventilation_Poor": [1 if ventilation == "Poor" else 0],
            "ventilation_Average": [1 if ventilation == "Average" else 0],
            "ventilation_Good": [1 if ventilation == "Good" else 0],
            "stress_Low": [1 if stress_level == "Low" else 0],
            "stress_Medium": [1 if stress_level == "Medium" else 0],
            "stress_High": [1 if stress_level == "High" else 0],
            "temperature": [temperature],
            "humidity": [humidity]
        })
        risk_score = model.predict(input_data)[0]
        tips = []
        if risk_score > 6:
            tips.append("Avoid outdoor activities from 6–10 AM due to high pollen levels.")
            tips.append("Wear a mask in high-traffic areas to reduce pollution exposure.")
        if ventilation == "Poor":
            tips.append("Use an air purifier or improve indoor ventilation.")
        if risk_score > 3 and medication_use == "None":
            tips.append("Consider using antihistamines; consult a doctor.")
        if not tips:
            tips.append("Low risk! Maintain current habits, but monitor pollen levels.")
        
        # Store results in session state
        st.session_state.results_shown = True
        st.session_state.risk_score = risk_score
        st.session_state.tips = tips
        st.session_state.chart_data = pd.DataFrame({
            "Factors": ["Pollen", "Air Quality", "Pollution", "Pets", "Diet", "Medication", "Ventilation", "Stress"],
            "Contribution": [0.3, 0.25, 0.15, 0.1, 0.08, 0.05, 0.04, 0.03]
        })

# Display results if available
if st.session_state.results_shown:
    st.subheader(f"Your Allergy Risk Score: {st.session_state.risk_score:.1f}/10")
    st.subheader("Daily Action Plan")
    for tip in st.session_state.tips:
        st.write(f"- {tip}")
    st.subheader("Risk Factors")
    st.bar_chart(st.session_state.chart_data.set_index("Factors"))

# Add a clear button to reset results
if st.session_state.results_shown:
    if st.button("Clear Results"):
        st.session_state.results_shown = False
        st.session_state.risk_score = None
        st.session_state.tips = []
        st.session_state.chart_data = None
        st.experimental_rerun()