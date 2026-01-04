import pandas as pd
import os

def fetch_data(csv_path):
    """Loads the property metadata and validates file existence."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Successfully fetched {len(df)} property records.")
    return df

if __name__ == "__main__":
    # Test the fetcher
    data = fetch_data("cleaned_dataset.csv")
    print(data.head())
