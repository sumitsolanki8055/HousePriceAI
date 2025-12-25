import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn.preprocessing import MinMaxScaler

# CONFIG
model_path = "house_price_model_v3.h5"
csv_path = "cleaned_dataset.csv"
img_folder = "data/house_images"  # Make sure this folder path matches your PC
cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'lat', 'long']

def get_img(path):
    if not os.path.exists(path): return None
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    return np.expand_dims(img / 255.0, axis=0)

def run_test():
    # Load
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except:
        print("Error: Model file not found.")
        return

    # Data
    df = pd.read_csv(csv_path)
    sc = MinMaxScaler()
    sc.fit(df[cols].values)
    
    # Random Batch
    batch = df.sample(10)
    print(f"{'ID':<12} | {'ACTUAL':<12} | {'PRED':<12} | {'DIFF':<10} | {'ERR %'}")
    print("-" * 75)

    total_diff = 0
    count = 0

    for _, row in batch.iterrows():
        hid = row['id']
        path = os.path.join(img_folder, f"image_{hid}.jpg")
        
        x_img = get_img(path)
        if x_img is None: continue

        x_stats = sc.transform(row[cols].values.reshape(1, -1))
        pred = model.predict([x_img, x_stats], verbose=0)[0][0]
        
        diff = abs(row['price'] - pred)
        err = (diff / row['price']) * 100
        total_diff += diff
        count += 1
        
        print(f"{str(hid):<12} | ${int(row['price']):<11,} | ${int(pred):<11,} | ${int(diff):<9,} | {err:.1f}%")

    if count > 0:
        print("-" * 75)
        print(f"AVG ERROR: ${int(total_diff/count):,}")

if __name__ == "__main__":
    run_test()
