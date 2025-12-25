import os
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CONFIG
input_file = "train(1).xlsx"
output_dir = "data/house_images"
delta = 0.0006
img_size = 600

def get_session():
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 429])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_bbox(lat, lng):
    return f"{lng - delta},{lat - delta},{lng + delta},{lat + delta}"

def fetch_image(session, lat, lng, pid):
    path = os.path.join(output_dir, f"image_{pid}.jpg")

    # Resume Logic
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return "SKIPPED"

    url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": get_bbox(lat, lng),
        "bboxSR": "4326",
        "size": f"{img_size},{img_size}",
        "f": "image",
        "format": "jpg"
    }

    try:
        res = session.get(url, params=params, stream=True, timeout=10)
        if res.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in res.iter_content(1024):
                    f.write(chunk)
            return "OK"
        elif res.status_code == 403:
            time.sleep(60)
            return "RATE_LIMIT"
        return "ERROR"
    except:
        return "ERROR"

def run_downloader():
    print("--- Starting Download Pipeline ---")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_excel(input_file, engine='openpyxl')
    existing = len(os.listdir(output_dir))
    
    print(f"Total Rows: {len(df)}")
    print(f"Existing Images: {existing}")
    print(f"Target: {len(df) - existing} new images")
    print("-" * 30)

    session = get_session()
    stats = {"OK": 0, "SKIPPED": 0, "ERROR": 0, "RATE_LIMIT": 0}

    for i, row in df.iterrows():
        try:
            lat, lng, pid = row.get('lat'), row.get('long'), row.get('id')
            if pd.isna(lat) or pd.isna(lng): continue

            status = fetch_image(session, lat, lng, pid)
            stats[status] += 1
            
            if status == "OK":
                if stats["OK"] % 50 == 0:
                    print(f"Downloaded {stats['OK']} new images...")
                time.sleep(0.2)
            elif status == "SKIPPED":
                if stats["SKIPPED"] % 2000 == 0:
                    print(f"Skipped {stats['SKIPPED']} existing files...")
            elif status == "RATE_LIMIT":
                print("Rate limit hit. Pausing 60s...")

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

    print("\n" + "="*30)
    print("Download Complete")
    print(f"New: {stats['OK']} | Skipped: {stats['SKIPPED']} | Errors: {stats['ERROR']}")
    print("="*30)

if __name__ == "__main__":
    run_downloader()
