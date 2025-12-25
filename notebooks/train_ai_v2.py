import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_FILE = "cleaned_dataset.csv"
IMAGE_FOLDER = "dataset/" 
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

# --- Load Data ---
df = pd.read_csv(CSV_FILE)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMAGE_FOLDER,
    x_col="filename",  # Update if column name differs
    y_col="price",     # Update if column name differs
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    subset="training"
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMAGE_FOLDER,
    x_col="filename",
    y_col="price",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    subset="validation"
)

# --- Build Model (v2) ---
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# --- Train ---
print("Starting training...")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# --- Save & Plot ---
model.save("house_price_model_v2.h5")
print("Model saved: house_price_model_v2.h5")

plt.plot(history.history['loss'], label='Training Error')
plt.plot(history.history['val_loss'], label='Validation Error')
plt.legend()
plt.show()
