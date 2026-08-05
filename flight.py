import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Flight Delay AI", page_icon="✈️", layout="wide")
st.title("✈️ Flight Delay Prediction System")
st.write("A Supervised Machine Learning model (Random Forest) that predicts if a flight will be delayed based on weather, time, and route.")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 2000
    
    airlines = np.random.choice(["IndiGo", "Air India", "SpiceJet", "Vistara"], n_samples)
    origins = np.random.choice(["Delhi (DEL)", "Mumbai (BOM)", "Bengaluru (BLR)", "Hyderabad (HYD)"], n_samples)
    destinations = np.random.choice(["Delhi (DEL)", "Mumbai (BOM)", "Bengaluru (BLR)", "Hyderabad (HYD)"], n_samples)
    weather = np.random.choice(["Clear", "Cloudy", "Rain", "Heavy Fog"], n_samples, p=[0.6, 0.25, 0.1, 0.05])
    hours = np.random.randint(0, 24, n_samples)
    
    delays = []
    for i in range(n_samples):
        delay_chance = 0.10 
        
        if weather[i] == "Heavy Fog": delay_chance += 0.70
        elif weather[i] == "Rain": delay_chance += 0.35
        
        if hours[i] >= 18: delay_chance += 0.20 
        if origins[i] == destinations[i]: delay_chance = 0.0 
            
        is_delayed = 1 if np.random.rand() < delay_chance else 0
        delays.append(is_delayed)
        
    df = pd.DataFrame({
        "Airline": airlines,
        "Origin": origins,
        "Destination": destinations,
        "Weather": weather,
        "Departure_Hour": hours,
        "Delayed": delays
    })
    
    df = df[df["Origin"] != df["Destination"]].reset_index(drop=True)
    
    encoders = {}
    for col in ["Airline", "Origin", "Destination", "Weather"]:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
        
    X = df.drop("Delayed", axis=1)
    y = df["Delayed"]
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X, y)
    
    return rf_model, encoders, df

model, label_encoders, raw_data = train_model()

st.sidebar.header("⚙️ Model Architecture")
st.sidebar.write("**Algorithm:** Random Forest Classifier")
st.sidebar.write("**Training Data:** 2,000 Historical Flights")
st.sidebar.write("**Features Used:** 5")
st.sidebar.write("---")
st.sidebar.write("*(Note: This uses synthetic data generated in-memory to ensure instant cloud deployment without needing external CSV files).*")

st.subheader("1. Enter Flight Details")
col1, col2 = st.columns(2)

with col1:
    selected_airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet", "Vistara"])
    selected_origin = st.selectbox("Origin City", ["Delhi (DEL)", "Mumbai (BOM)", "Bengaluru (BLR)", "Hyderabad (HYD)"])
    selected_dest = st.selectbox("Destination City", ["Mumbai (BOM)", "Delhi (DEL)", "Bengaluru (BLR)", "Hyderabad (HYD)"])

with col2:
    selected_weather = st.selectbox("Weather Conditions", ["Clear", "Cloudy", "Rain", "Heavy Fog"])
    selected_hour = st.slider("Departure Time (24-Hour Format)", 0, 23, 12)

st.subheader("2. AI Prediction")

if selected_origin == selected_dest:
    st.error("Origin and Destination cannot be the same city.")
else:
    if st.button("🔮 Predict Delay Probability", type="primary"):
        with st.spinner("Analyzing historical data..."):
            
            input_data = {
                "Airline": label_encoders["Airline"].transform([selected_airline])[0],
                "Origin": label_encoders["Origin"].transform([selected_origin])[0],
                "Destination": label_encoders["Destination"].transform([selected_dest])[0],
                "Weather": label_encoders["Weather"].transform([selected_weather])[0],
                "Departure_Hour": selected_hour
            }
            
            input_df = pd.DataFrame([input_data])
            
            prediction_prob = model.predict_proba(input_df)[0]
            delay_probability = prediction_prob[1] * 100
            
            if delay_probability > 50:
                st.error(f"⚠️ **High Risk of Delay:** The AI predicts a {delay_probability:.1f}% chance this flight will be delayed.")
                st.progress(int(delay_probability))
            elif delay_probability > 25:
                st.warning(f"⛅ **Moderate Risk of Delay:** The AI predicts a {delay_probability:.1f}% chance this flight will be delayed.")
                st.progress(int(delay_probability))
            else:
                st.success(f"✅ **Likely On Time:** The AI predicts only a {delay_probability:.1f}% chance of delay.")
                st.progress(int(delay_probability))
