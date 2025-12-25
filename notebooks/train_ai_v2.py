import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# --- 1. Configuration & Setup ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "dataset/house_images"  # Example directory

print("--- Loading Data ---")
# Note: This assumes you have a function or data generator to load images
# For this script, we simulate the structure matching your logs:
# Loading 12892 images (approx 10,300 for train, 2,500 for validation with 20% split)

# Use image_dataset_from_directory for efficient loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int' # Use 'int' or 'float' depending on your label CSV
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Normalize pixel values to be between 0 and 1
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

print(f"Loading {12892} images at {IMG_HEIGHT}x{IMG_WIDTH}...\n")


# --- 2. Building the "Advanced" Model (v2) ---
print("--- Building Advanced Model ---")

model = models.Sequential([
    # Input Layer
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Block 2 (The "Advanced" part usually adds depth)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening for Dense Layers
    layers.Flatten(),
    layers.Dropout(0.5), # Helps prevent overfitting
    
    # Dense Layers for Regression
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    
    # Output Layer: 1 neuron, no activation (linear) for predicting price
    layers.Dense(1) 
])

# Using Mean Absolute Error (MAE) because the logs showed error in dollars ($150,000)
model.compile(optimizer='adam',
              loss='mean_absolute_error', 
              metrics=['mean_absolute_error'])

print("\n--- Starting Training (30 Epochs) ---")


# --- 3. Training ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# --- 4. Evaluation & Results ---
print("\n--- Evaluation ---")
# We assume the "Original Model Error" was a hardcoded benchmark
original_error = 225000.00
final_loss = history.history['val_loss'][-1]

print(f"Original Model Error: ~${original_error:,.2f}")
print(f"New Model Error:      ${final_loss:,.2f}\n")


# --- 5. Plotting (The Graph) ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Error (Price Prediction)')
plt.ylabel('Error ($)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show() # This generates the image in the notebook


# --- 6. Saving ---
# Saving as .h5 as seen in your logs (despite the legacy warning)
model.save('house_price_model_v2.h5')
print("Model saved as house_price_model_v2.h5")
