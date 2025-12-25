# 🏡 AI House Price Predictor

A Multi-Modal Deep Learning application that predicts real estate prices by analyzing both **visual data** (satellite imagery) and **numerical data** (bedrooms, bathrooms, sqft) simultaneously.

## 🚀 Features
* **Multi-Modal Brain:** Uses a custom Neural Network that combines a CNN (for image processing) with a Dense Network (for statistics).
* **Interactive UI:** Built with **Streamlit** for real-time predictions.
* **High Accuracy:** Fine-tuned "V3" model achieves ~6% error margin on test data.
* **Visual Analysis:** The AI actually "looks" at the house photo to judge value (e.g., luxury vs. simple structures).

## 🛠️ Tech Stack
* **Python 3.10+**
* **TensorFlow / Keras** (Deep Learning)
* **Streamlit** (Web Interface)
* **Pandas & NumPy** (Data Processing)
* **OpenCV & PIL** (Image Processing)

## 📂 Project Structure
```text
HousePriceAI/
├── app.py                   # The main Streamlit web application
├── house_price_model_v3.h5  # The trained AI Brain (The Model)
├── cleaned_dataset.csv      # Processed data for scaling
├── sample_images/           # Test images for you to try
│   ├── luxury_house.jpg
│   └── small_house.jpg
└── README.md                # This file