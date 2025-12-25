import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

old_model = "house_price_model_v2.h5"
new_model = "house_price_model_v3.h5"
csv_path = "cleaned_dataset.csv"
img_folder = "data/house_images"
size = 128
cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'lat', 'long']

def load_data():
    print("--- Loading Data ---")
    df = pd.read_csv(csv_path)
    
    df['path'] = df['id'].apply(lambda x: os.path.join(img_folder, f"image_{x}.jpg"))
    df = df[df['path'].apply(os.path.exists)]
    
    print(f"Images found: {len(df)}")
    
    images = []
    for path in df['path']:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        images.append(img)
    
    x_img = np.array(images) / 255.0
    
    sc = MinMaxScaler()
    x_stats = sc.fit_transform(df[cols].values)
    
    y = df['price'].values
    return x_img, x_stats, y

def run_tuning():
    x_img, x_stats, y = load_data()
    
    xi_train, xi_test, xs_train, xs_test, y_train, y_test = train_test_split(
        x_img, x_stats, y, test_size=0.2, random_state=42
    )

    print(f"\n--- Loading {old_model} ---")
    model = tf.keras.models.load_model(old_model, compile=False)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    print("\n--- Fine-Tuning ---")
    history = model.fit(
        [xi_train, xs_train], y_train,
        validation_data=([xi_test, xs_test], y_test),
        epochs=15,
        batch_size=32
    )

    model.save(new_model)
    print(f"\nSaved: {new_model}")

    final_loss = history.history['val_loss'][-1]
    print(f"Final Val Loss: {int(final_loss):,}")
    print(f"RMSE: ${int(np.sqrt(final_loss)):,}")

if __name__ == "__main__":
    run_tuning()
