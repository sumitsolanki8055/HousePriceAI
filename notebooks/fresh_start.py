import os
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CONFIGURATION
INPUT_FILE = "train(1).xlsx"
OUTPUT_DIR = "data/house_images"
DELTA = 0.0006  # Zoom level
IMG_SIZE = 600

def get_session():
    """Creates a resilient session with auto-retries."""
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 429])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_bbox(lat, lng):
    return f"{lng - DELTA},{lat - DELTA},{lng + DELTA},{lat + DELTA}"

def fetch_image(session, lat, long, pid):
    path = os.path.join(OUTPUT_DIR, f"image_{pid}.jpg")

    # Resume Logic: Skip if already downloaded
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return

    url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": get_bbox(lat, long),
        "bboxSR": "4326",
        "size": f"{IMG_SIZE},{IMG_SIZE}",
        "f": "image",
        "format": "jpg"
    }

    try:
        res = session.get(url, params=params, stream=True, timeout=10)
        if res.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in res.iter_content(1024):
                    f.write(chunk)
            print(f"[OK] {pid}")
        elif res.status_code == 403:
            print("[PAUSE] Rate limited. Sleeping 60s...")
            time.sleep(60)
        else:
            print(f"[ERR] {pid}: HTTP {res.status_code}")
    except Exception as e:
        print(f"[FAIL] {pid}: {e}")

def run_downloader():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"--- Loading {INPUT_FILE} ---")
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    
    # Normalize column names just in case
    df.columns = [c.lower() for c in df.columns]
    
    if 'price' not in df.columns:
        print("Error: Dataset missing 'price' column.")
        return

    print(f"Starting download for {len(df)} properties...")
    session = get_session()

    for i, row in df.iterrows():
        try:
            lat = row.get('lat') or row.get('latitude')
            lng = row.get('long') or row.get('longitude')
            pid = row.get('id', i)

            if pd.isna(lat) or pd.isna(lng): continue
            
            fetch_image(session, lat, lng, pid)
            time.sleep(0.2) # Be polite to the server

        except KeyboardInterrupt:
            print("\nStopped by user
