import os
import pandas as pd
from PIL import Image, UnidentifiedImageError

def clean_csv_by_images(csv_path, images_folder):
    # Load CSV
    df = pd.read_csv(csv_path)
    # Assume image filenames are sample_id.jpg
    bad_ids = []
    for idx, row in df.iterrows():
        sample_id = str(row['sample_id'])
        # Try jpg and png extensions
        found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(images_folder, f"{sample_id}{ext}")
            if os.path.exists(img_path):
                found = True
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except UnidentifiedImageError:
                    print(f"Unidentified image: {img_path}")
                    bad_ids.append(row['sample_id'])
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")
                break
        if not found:
            print(f"Image not found for sample_id {sample_id}")
            bad_ids.append(row['sample_id'])
    # Remove bad rows
    cleaned_df = df[~df['sample_id'].isin(bad_ids)]
    cleaned_df.to_csv(csv_path, index=False)
    print(f"Removed {len(bad_ids)} bad images from {csv_path}")

if __name__ == "__main__":
    clean_csv_by_images(
        "/home/arnavw/Documents/amazon-ml-2025/dataset/train_split/train_part1.csv",
        "/home/arnavw/Documents/amazon-ml-2025/images/train_part1"
    )
    clean_csv_by_images(
        "/home/arnavw/Documents/amazon-ml-2025/dataset/val_split/val_part1.csv",
        "/home/arnavw/Documents/amazon-ml-2025/images/val_part1"
    )