"""
History Logging Utility Module.

Provides functions to log inference metadata to a CSV file and save the 
corresponding output images to the local filesystem for persistent storage.
"""

import os
import csv
import datetime
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import numpy as np


def log_translation(filename: str, width: int, height: int, inference_time_ms: float, edge_clarity: float, output_img_array: np.ndarray) -> str:
    """
    Logs translation metadata to history/history_log.csv and saves the output image
    under history/outputs/.

    Args:
        filename (str): Name of the uploaded file.
        width (int): Image width.
        height (int): Image height.
        inference_time_ms (float): Execution time in milliseconds.
        edge_clarity (float): CLAHE limit parameter.
        output_img_array (np.ndarray): Translated image array (RGB).

    Returns:
        str: Relative path to the saved output image.
    """
    history_dir = "history"
    outputs_dir = os.path.join(history_dir, "outputs")
    
    # Ensure directories exist
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Generate unique output filename
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_filename = f"{base_name}_{timestamp_str}.png"
    output_path = os.path.join(outputs_dir, output_filename)
    
    # Save the output image
    output_pil = Image.fromarray(output_img_array)
    output_pil.save(output_path)
    
    # Log details to CSV
    csv_path = os.path.join(history_dir, "history_log.csv")
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header if file is new
            writer.writerow([
                "Timestamp", 
                "Original Filename", 
                "Width", 
                "Height", 
                "Inference Time (ms)", 
                "CLAHE Limit", 
                "Output Image Path"
            ])
        
        # Write log entry
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            width,
            height,
            f"{inference_time_ms:.2f}",
            f"{edge_clarity:.1f}",
            output_path
        ])
        
    return output_path
