import os
import pandas as pd

# CONFIG
input_file = "train(1).xlsx"
output_file = "cleaned_dataset.csv"
img_dir = "data/house_images"

def match_data():
    print(f"--- Processing {input_file} ---")
    
    # 1. Load Excel
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    df = pd.read_excel(input_file, engine='openpyxl')
    total_rows = len(df)
    
    # 2. Get list of valid images
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} does not exist.")
        return
        
    valid_images = set(os.listdir(img_dir))
    print(f"Images found on disk: {len(valid_images)}")

    # 3. Filter DataFrame
    # We create a temporary column to check existence
    df['img_exists'] = df['id'].apply(lambda x: f"image_{x}.jpg" in valid_images)
    
    # Keep only rows where img_exists is True
    clean_df = df[df['img_exists']].copy()
    clean_df.drop(columns=['img_exists'], inplace=True)
    
    # 4. Save
    clean_df.to_csv(output_file, index=False)
    
    # 5. Report
    print("-" * 30)
    print(f"Original Rows: {total_rows}")
    print(f"Cleaned Rows:  {len(clean_df)}")
    print(f"Dropped Rows:  {total_rows - len(clean_df)}")
    print("-" * 30)
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    match_data()
