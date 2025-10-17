import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = pickle.load(open("model/model_final.pkl", "rb"))

# Set page config
st.set_page_config(page_title="🌸 Iris Flower Predictor", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1 {
            color: #6a1b9a;
        }
        .stButton>button {
            background-color: #6a1b9a;
            color: white;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("🌸 Iris Flower Species Prediction")
st.markdown("Use the sliders below to enter flower measurements and predict the species 🌼")

# Form layout using columns
with st.form("iris_app_form"):
    col1, col2 = st.columns(2)

    with col1:
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)

    with col2:
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
        pw = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

    # Submit button
    submitted = st.form_submit_button("🔍 Predict Species")

# On submit
if submitted:
    # Prepare input for prediction
    input_data = np.array([[sl, pl, sw, pw]])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Mapping output to species
    species_mapping = {0: "Setosa 🌱", 1: "Versicolor 🌸", 2: "Virginica 🌺"}
    species_name = species_mapping.get(prediction[0], "Unknown")

    # Display prediction
    st.markdown(f"### ✅ Predicted Species: **{species_name}**")

    # Display confidence
    confidence = np.max(prediction_proba) * 100
    st.markdown(f"**Prediction Confidence:** `{confidence:.2f}%`")

    # Show probability bar chart
    species_labels = [species_mapping[i] for i in range(3)]
    probabilities = prediction_proba[0]

    df = pd.DataFrame({
        "Species": species_labels,
        "Probability": probabilities
    })

    st.subheader("📊 Probability Distribution")
    st.bar_chart(df.set_index("Species"))

    # Add a fun touch
    st.info("The model is trained on the classic Iris dataset. Enjoy exploring!")
