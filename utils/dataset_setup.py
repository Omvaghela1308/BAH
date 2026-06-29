"""
Dataset Setup and Synthetic Generator Utility.

Creates the standard dataset directory structure (data/train/ir, data/train/rgb,
data/val/ir, data/val/rgb) and generates high-fidelity simulated/synthetic
satellite images. This allows immediate testing of the training pipeline.
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from PIL import Image


def create_directories():
    """Create the training and validation dataset directories."""
    dirs = [
        "data/train/ir",
        "data/train/rgb",
        "data/val/ir",
        "data/val/rgb"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Dataset directories created successfully.")


def generate_synthetic_satellite_pair(index: int, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a high-fidelity synthetic pair representing a satellite view.
    
    Includes simulated terrain features like oceans, landmasses, rivers, forests,
    and cloud patches, producing both a realistic RGB image and its matching IR representation.
    """
    # Create canvas
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    ir = np.zeros((size, size), dtype=np.uint8)
    
    # 1. Base Ocean (Deep Blue for RGB, Low thermal/near-IR reflectance for IR)
    rgb[:, :] = [10, 30, 80]
    ir[:, :] = 25  # Water is dark in NIR/thermal
    
    # Use random seeds based on index to create unique terrain layouts
    np.random.seed(index * 12345)
    
    # 2. Add Landmasses (Green/brown vegetation in RGB, highly reflective/bright in NIR/thermal IR)
    # Generate random shapes using blobs
    grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
    
    # Multiple center points for landmasses
    num_islands = np.random.randint(2, 5)
    for _ in range(num_islands):
        cx = np.random.randint(30, size - 30)
        cy = np.random.randint(30, size - 30)
        radius = np.random.randint(40, 100)
        
        # Calculate distance with some noise to make island edges natural
        noise = np.sin(grid_x / 10.0) * 10 + np.cos(grid_y / 10.0) * 10
        dist = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2) + noise
        
        mask = dist < radius
        
        # RGB Land color (warm earthy greens/browns)
        land_color = [np.random.randint(34, 70), np.random.randint(100, 150), np.random.randint(34, 70)]
        rgb[mask] = land_color
        
        # IR Land (Plants are highly reflective in Near-IR, so bright IR signature)
        ir[mask] = np.random.randint(150, 220)

    # 3. Add Rivers (Sinuous curves in RGB, dark in IR)
    num_rivers = np.random.randint(1, 3)
    for _ in range(num_rivers):
        # Generate simple bezier/sin curve for river
        start_x = np.random.randint(0, size)
        start_y = 0
        end_x = np.random.randint(0, size)
        end_y = size - 1
        
        # Sinuous path
        t = np.linspace(0, 1, size)
        river_x = (1 - t) * start_x + t * end_x + np.sin(t * 10) * 15
        river_x = np.clip(river_x, 0, size - 1).astype(int)
        
        for y_idx in range(size):
            rx = river_x[y_idx]
            # River width
            w = np.random.randint(2, 5)
            left = max(0, rx - w)
            right = min(size, rx + w + 1)
            
            rgb[y_idx, left:right] = [12, 45, 95]
            ir[y_idx, left:right] = 20

    # 4. Add Clouds (Translucent White in RGB, Bright white/cold in IR)
    num_clouds = np.random.randint(1, 4)
    for _ in range(num_clouds):
        cx = np.random.randint(0, size)
        cy = np.random.randint(0, size)
        cloud_size = np.random.randint(20, 60)
        
        cloud_mask = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2) < cloud_size
        
        # Soften edges
        if np.any(cloud_mask):
            blurred_mask = cv2.GaussianBlur(cloud_mask.astype(np.float32), (21, 21), 0)
            for c in range(3):
                rgb[:, :, c] = (rgb[:, :, c] * (1 - blurred_mask) + 245 * blurred_mask).astype(np.uint8)
            ir = (ir * (1 - blurred_mask) + 230 * blurred_mask).astype(np.uint8)
            
    # Add a bit of sensor noise to look realistic
    rgb_noise = np.random.normal(0, 3, rgb.shape).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16) + rgb_noise, 0, 255).astype(np.uint8)
    
    ir_noise = np.random.normal(0, 2, ir.shape).astype(np.int16)
    ir = np.clip(ir.astype(np.int16) + ir_noise, 0, 255).astype(np.uint8)

    return ir, rgb


def generate_dataset(num_train: int = 50, num_val: int = 10):
    """Generate the synthetic dataset files."""
    create_directories()
    
    # Generate Training Data
    print(f"Generating {num_train} synthetic training pairs...")
    for idx in range(num_train):
        ir, rgb = generate_synthetic_satellite_pair(idx)
        
        ir_filename = f"sat_image_{idx:04d}.png"
        rgb_filename = f"sat_image_{idx:04d}.png"  # filenames must match for dataset loader
        
        Image.fromarray(ir).save(os.path.join("data/train/ir", ir_filename))
        Image.fromarray(rgb).save(os.path.join("data/train/rgb", rgb_filename))
        
    # Generate Validation Data
    print(f"Generating {num_val} synthetic validation pairs...")
    for idx in range(num_val):
        # Offset seed to make validation data unique
        ir, rgb = generate_synthetic_satellite_pair(idx + 1000)
        
        ir_filename = f"sat_image_val_{idx:04d}.png"
        rgb_filename = f"sat_image_val_{idx:04d}.png"
        
        Image.fromarray(ir).save(os.path.join("data/val/ir", ir_filename))
        Image.fromarray(rgb).save(os.path.join("data/val/rgb", rgb_filename))
        
    print("\n--- Setup Complete ---")
    print(f"Dataset generated at: {os.path.abspath('data')}")
    print("\nTo integrate real-world datasets:")
    print("1. Collect your paired grayscale Infrared (IR) and Optical (RGB) images.")
    print("2. Drop the IR images into 'data/train/ir' and matching RGB images into 'data/train/rgb' with identical names.")
    print("3. Ensure validation images are placed in 'data/val/ir' and 'data/val/rgb' similarly.")
    print("4. The training pipeline will automatically scan and align files with matching names.")


if __name__ == "__main__":
    generate_dataset()
