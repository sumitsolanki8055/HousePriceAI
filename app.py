import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import os

# --- PATH CONFIGURATION ---
MODEL_PATH = "house_price_model_v3.h5" 
DATA_FILE = "cleaned_dataset.csv"
IMG_SIZE = 128

@st.cache_resource
def load_data_and_scaler():
    if not os.path.exists(DATA_FILE):
        st.error("❌ Error: DATA_FILE not found.")
        st.stop()
    df = pd.read_csv(DATA_FILE)
    feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'lat', 'long']
    scaler = MinMaxScaler()
    scaler.fit(df[feature_cols].values)
    return scaler

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Error: MODEL_PATH not found.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

st.title("🏡 AI House Price Predictor")

try:
    scaler = load_data_and_scaler()
    model = load_model()
    st.success("Brain Loaded! Ready to predict.")
except Exception as e:
    st.error(f"Something went wrong: {e}")

# INPUTS
bed = st.slider("Bedrooms", 1, 10, 3)
bath = st.slider("Bathrooms", 1, 10, 2)
sqft = st.number_input("Square Feet", value=2000)
lat, long = 47.6, -122.3 # Defaults

uploaded_file = st.file_uploader("Upload House Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file and st.button("Predict Price"):
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Your House", use_column_width=True)
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    stats = np.array([[bed, bath, sqft, 5000, 1, lat, long]])
    stats_scaled = scaler.transform(stats)
    
    price = model.predict([img_array, stats_scaled])[0][0]
    st.header(f"💰 Predicted Price: ${int(price):,}")