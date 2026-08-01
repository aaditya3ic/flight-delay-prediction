# ==============================================================================
# PROJECT 5: FLIGHT DELAY PREDICTION (Supervised Machine Learning)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------------------
# STEP 1: Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Flight Delay AI", page_icon="✈️", layout="wide")
st.title("✈️ Flight Delay Prediction System")
st.write("A Supervised Machine Learning model (Random Forest) that predicts if a flight will be delayed based on weather, time, and route.")

# ------------------------------------------------------------------------------
# STEP 2: Generate Training Data & Train Model (Cached for Performance)
# ------------------------------------------------------------------------------
@st.cache_resource
def train_model():
    # 1. Generate realistic synthetic data for the model to learn from
    np.random.seed(42)
    n_samples = 2000
    
    airlines = np.random.choice(["IndiGo", "Air India", "SpiceJet", "Vistara"], n_samples)
    origins = np.random.choice(["Delhi (DEL)", "Mumbai (BOM)", "Bengaluru (BLR)", "Hyderabad (HYD)"], n_samples)
    destinations = np.random.choice(["Delhi (DEL)", "Mumbai (BOM)", "Bengaluru (BLR)", "Hyderabad (HYD)"], n_samples)
    weather = np.random.choice(["Clear", "Cloudy", "Rain", "Heavy Fog"], n_samples, p=[0.6, 0.25, 0.1, 0.05])
    hours = np.random.randint(0, 24, n_samples)
    
    # 2. Define the logical rules for what causes a delay (The model will learn these automatically)
    delays = []
    for i in range(n_samples):
        delay_chance = 0.10 # Base 10% chance
        
        if weather[i] == "Heavy Fog": delay_chance += 0.70
        elif weather[i] == "Rain": delay_chance += 0.35
        
        if hours[i] >= 18: delay_chance += 0.20 # Evening rush hour delays
        if origins[i] == destinations[i]: delay_chance = 0.0 # Same city = no flight
            
        # Determine final status based on probability
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
    
    # Remove rows where Origin == Destination
    df = df[df["Origin"] != df["Destination"]].reset_index(drop=True)
    
    # 3. Preprocess Data: Convert text labels (Strings) into Numbers (Integers) for the ML model
    encoders = {}
    for col in ["Airline", "Origin", "Destination", "Weather"]:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
        
    # 4. Train the Random Forest Model
    X = df.drop("Delayed", axis=1)
    y = df["Delayed"]
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X, y)
    
    return rf_model, encoders, df

model, label_encoders, raw_data = train_model()

# ------------------------------------------------------------------------------
# STEP 3: User Interface for Inputs
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# STEP 4: AI Prediction Logic
# ------------------------------------------------------------------------------
st.subheader("2. AI Prediction")

if selected_origin == selected_dest:
    st.error("Origin and Destination cannot be the same city.")
else:
    if st.button("🔮 Predict Delay Probability", type="primary"):
        with st.spinner("Analyzing historical data..."):
            
            # Encode user inputs exactly how the model was trained
            input_data = {
                "Airline": label_encoders["Airline"].transform([selected_airline])[0],
                "Origin": label_encoders["Origin"].transform([selected_origin])[0],
                "Destination": label_encoders["Destination"].transform([selected_dest])[0],
                "Weather": label_encoders["Weather"].transform([selected_weather])[0],
                "Departure_Hour": selected_hour
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Get Probability (Returns array like [[% On Time, % Delayed]])
            prediction_prob = model.predict_proba(input_df)[0]
            delay_probability = prediction_prob[1] * 100
            
            # Display Results
            if delay_probability > 50:
                st.error(f"⚠️ **High Risk of Delay:** The AI predicts a {delay_probability:.1f}% chance this flight will be delayed.")
                st.progress(int(delay_probability))
            elif delay_probability > 25:
                st.warning(f"⛅ **Moderate Risk of Delay:** The AI predicts a {delay_probability:.1f}% chance this flight will be delayed.")
                st.progress(int(delay_probability))
            else:
                st.success(f"✅ **Likely On Time:** The AI predicts only a {delay_probability:.1f}% chance of delay.")
                st.progress(int(delay_probability))

# ------------------------------------------------------------------------------
# STEP 5: Educational Resources
# ------------------------------------------------------------------------------
st.markdown("---")
st.header("📚 Presentation Talking Points")

with st.expander("🛠️ How it works: Supervised Learning & Random Forests"):
    st.write("""
    - **Supervised Learning:** We gave the algorithm thousands of past flights alongside the *answers* (whether they were delayed or not). It analyzed this data to find patterns.
    - **Random Forest:** Instead of using one decision tree, the AI builds 100 different decision trees and lets them 'vote' on the final outcome. This prevents the model from jumping to bad conclusions based on outliers.
    - **Label Encoding:** Machine learning models only understand math. We had to convert text labels (like 'Clear' or 'Rain') into numerical IDs (like 0 or 2) before feeding them into the network.
    """)

with st.expander("🎯 Student Challenge"):
    st.write("Play with the inputs to reverse-engineer what the AI learned. Discuss as a class:")
    st.markdown("""
    1. Set the weather to **Clear** at **10 AM**. What is the delay probability?
    2. Change the weather to **Heavy Fog**. How drastically does the probability jump?
    3. Change the time to **9 PM (21:00)** during the Rain. Why do delays cascade later in the day?
    
    *This illustrates feature importance: the model figured out entirely on its own that Weather and Time are much stronger predictors of a delay than the Airline name.*
    """)

with st.expander("💼 Interview Corner"):
    st.markdown("""
    1. **Classification vs. Regression:** Why is predicting 'Delayed vs. On Time' a classification problem, but predicting 'Minutes of Delay' a regression problem?
    2. **Overfitting:** What happens if a Random Forest tree gets too deep? (Hint: It memorizes the training data instead of learning general patterns).
    3. **Imbalanced Data:** If 90% of flights are on time, a "dumb" model could just guess "On Time" every time and be 90% accurate. How do we properly measure the success of models with imbalanced targets?
    """)