"""
Dataset Module for Satellite Image Translation.

Provides a PyTorch Dataset class designed to load paired Infrared (IR) and
Optical RGB images for model training and validation.
"""

import os
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from torch.utils.data import Dataset
# pyrefly: ignore [missing-import]
from PIL import Image


class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for loading paired satellite images.
    
    Expects two directories: one containing infrared (IR) grayscale images,
    and another containing the corresponding optical (RGB) images with matching filenames.
    """
    def __init__(self, ir_dir: str, rgb_dir: str, image_size: int = 256) -> None:
        """
        Initialize the dataset.

        Args:
            ir_dir (str): Path to directory with infrared (IR) images.
            rgb_dir (str): Path to directory with optical (RGB) images.
            image_size (int): Size to resize images to (default: 256).
        """
        self.ir_dir = ir_dir
        self.rgb_dir = rgb_dir
        self.image_size = image_size
        
        # Verify directories exist
        if not os.path.exists(ir_dir):
            raise FileNotFoundError(f"Infrared directory not found: {ir_dir}")
        if not os.path.exists(rgb_dir):
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
            
        # Get list of filenames that exist in both directories
        ir_files = set(os.listdir(ir_dir))
        rgb_files = set(os.listdir(rgb_dir))
        self.filenames = sorted(list(ir_files.intersection(rgb_files)))
        
        if len(self.filenames) == 0:
            print(f"Warning: No overlapping filenames found between {ir_dir} and {rgb_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the preprocessed IR and RGB image pair at the given index.

        Args:
            idx (int): Index of the image pair to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - ir_tensor: Preprocessed IR image of shape [1, 256, 256]
                - rgb_tensor: Preprocessed RGB image of shape [3, 256, 256]
        """
        filename = self.filenames[idx]
        ir_path = os.path.join(self.ir_dir, filename)
        rgb_path = os.path.join(self.rgb_dir, filename)

        # Load IR image
        ir_pil = Image.open(ir_path)
        ir_np = np.array(ir_pil)

        # Convert to grayscale
        if len(ir_np.shape) == 3:
            if ir_np.shape[2] == 3:
                ir_gray = cv2.cvtColor(ir_np, cv2.COLOR_RGB2GRAY)
            elif ir_np.shape[2] == 4:
                ir_gray = cv2.cvtColor(ir_np, cv2.COLOR_RGBA2GRAY)
            else:
                ir_gray = ir_np
        else:
            ir_gray = ir_np

        # Resize to image_size
        ir_resized = cv2.resize(ir_gray, (self.image_size, self.image_size))
        
        # Normalize to [-1.0, 1.0]
        ir_normalized = (ir_resized.astype(np.float32) / 127.5) - 1.0
        
        # Convert to single-channel tensor: [1, H, W]
        ir_tensor = torch.from_numpy(ir_normalized).unsqueeze(0)

        # Load RGB image
        rgb_pil = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb_pil)
        
        # Resize to image_size
        rgb_resized = cv2.resize(rgb_np, (self.image_size, self.image_size))
        
        # Normalize to [-1.0, 1.0] (matching model Tanh output range)
        rgb_normalized = (rgb_resized.astype(np.float32) / 127.5) - 1.0
        
        # Convert to 3-channel tensor: [3, H, W]
        rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1)

        return ir_tensor, rgb_tensor
