"""
Landsat 8 Dataset Downloader and Preprocessor.

Downloads Landsat 8 bands (B5 for IR; B4, B3, B2 for RGB) for a sample scene
from the public Google Cloud Storage bucket, crops the central region, normalizes
the 16-bit TIFF values to standard 8-bit imagery, and slices them into 256x256 patches.
"""

import os
import urllib.request
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image


def download_file_with_progress(url: str, output_path: str):
    """Download a file showing progress in the console."""
    print(f"Downloading: {url}")
    start_time = time.time()
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            elapsed = time.time() - start_time
            speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            print(f"\rProgress: {percent:.1f}% | {downloaded / (1024 * 1024):.1f} MB of {total_size / (1024 * 1024):.1f} MB | Speed: {speed:.2f} MB/s", end="")
        else:
            print(f"\rProgress: {downloaded / (1024 * 1024):.1f} MB downloaded", end="")

    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print(f"\nFinished download. Saved to {output_path}\n")


def process_landsat_scene():
    # Public URL path for a sample Landsat 8 scene on GCS (San Francisco Bay area)
    base_url = "https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/044/034/LC08_L1TP_044034_20130603_20170310_01_T1"
    scene_id = "LC08_L1TP_044034_20130603_20170310_01_T1"
    
    cache_dir = "data/landsat_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # We need bands: B5 (Near-IR), B4 (Red), B3 (Green), B2 (Blue)
    bands = ["B5", "B4", "B3", "B2"]
    band_paths = {}
    
    # Step 1: Download bands
    for band in bands:
        url = f"{base_url}/{scene_id}_{band}.TIF"
        dest_path = os.path.join(cache_dir, f"{band}.TIF")
        band_paths[band] = dest_path
        
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 10 * 1024 * 1024:
            try:
                download_file_with_progress(url, dest_path)
            except Exception as e:
                print(f"Error downloading {band}: {e}")
                print("Make sure you are connected to the internet.")
                return
        else:
            print(f"Band {band} already downloaded.")

    # Step 2: Open and Crop
    print("Processing and aligning Landsat 8 bands...")
    
    # Open bands (Landsat 8 bands are 16-bit GeoTIFFs)
    img_b5 = Image.open(band_paths["B5"])  # NIR
    img_b4 = Image.open(band_paths["B4"])  # Red
    img_b3 = Image.open(band_paths["B3"])  # Green
    img_b2 = Image.open(band_paths["B2"])  # Blue
    
    w, h = img_b5.size
    print(f"Raw scene dimensions: {w}x{h}")
    
    # Define center crop for training (1024x1024)
    # SF Bay/city center area of the scene
    cx, cy = w // 2 - 400, h // 2 - 400
    crop_size = 1024
    
    box_train = (cx - crop_size // 2, cy - crop_size // 2, cx + crop_size // 2, cy + crop_size // 2)
    # Define crop for validation (512x512 offset from center)
    box_val = (cx - crop_size // 2 - 600, cy - crop_size // 2 - 600, cx - crop_size // 2 - 88, cy - crop_size // 2 - 88)
    
    # Create final training/val dirs
    os.makedirs("data/train/ir", exist_ok=True)
    os.makedirs("data/train/rgb", exist_ok=True)
    os.makedirs("data/val/ir", exist_ok=True)
    os.makedirs("data/val/rgb", exist_ok=True)
    
    def process_and_slice(box, subset_name):
        print(f"\nProcessing {subset_name} subset...")
        
        # Crop 16-bit bands
        b5_crop = np.array(img_b5.crop(box), dtype=np.float32)
        b4_crop = np.array(img_b4.crop(box), dtype=np.float32)
        b3_crop = np.array(img_b3.crop(box), dtype=np.float32)
        b2_crop = np.array(img_b2.crop(box), dtype=np.float32)
        
        # Dynamic normalization using 99.5th percentile to scale 16-bit to 8-bit
        # Helps ignore specular reflections or clouds and makes imagery high-contrast
        scale_b5 = np.percentile(b5_crop, 99.5)
        scale_b4 = np.percentile(b4_crop, 99.5)
        scale_b3 = np.percentile(b3_crop, 99.5)
        scale_b2 = np.percentile(b2_crop, 99.5)
        
        # Scale and clip to [0, 255] uint8
        b5_8bit = np.clip((b5_crop / scale_b5) * 255.0, 0, 255).astype(np.uint8)
        b4_8bit = np.clip((b4_crop / scale_b4) * 255.0, 0, 255).astype(np.uint8)
        b3_8bit = np.clip((b3_crop / scale_b3) * 255.0, 0, 255).astype(np.uint8)
        b2_8bit = np.clip((b2_crop / scale_b2) * 255.0, 0, 255).astype(np.uint8)
        
        # Stack channels to create RGB
        rgb_8bit = np.stack([b4_8bit, b3_8bit, b2_8bit], axis=-1)
        
        # Slice into 256x256 patches
        crop_h, crop_w = b5_crop.shape
        patch_size = 256
        patch_count = 0
        
        for y in range(0, crop_h, patch_size):
            for x in range(0, crop_w, patch_size):
                if y + patch_size <= crop_h and x + patch_size <= crop_w:
                    ir_patch = b5_8bit[y:y+patch_size, x:x+patch_size]
                    rgb_patch = rgb_8bit[y:y+patch_size, x:x+patch_size]
                    
                    # Filename
                    filename = f"landsat_{subset_name}_{patch_count:03d}.png"
                    
                    # Save patches
                    Image.fromarray(ir_patch).save(os.path.join(f"data/{subset_name}/ir", filename))
                    Image.fromarray(rgb_patch).save(os.path.join(f"data/{subset_name}/rgb", filename))
                    
                    patch_count += 1
                    
        print(f"Generated {patch_count} patches for {subset_name}.")

    process_and_slice(box_train, "train")
    process_and_slice(box_val, "val")
    print("\n--- Landsat 8 Processing Complete ---")


if __name__ == "__main__":
    process_landsat_scene()
