"""
Transforms and Inference Module for Satellite Image Translation.

Provides a set of helper functions to preprocess raw input PIL images into single-channel
tensors, run the model translation forward pass, and format outputs into standard
visual numpy arrays.
"""

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from PIL import Image


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image into a grayscale PyTorch tensor ready for model inference.

    Conversion steps:
    1. Convert PIL image to an OpenCV numpy array.
    2. Convert color array to grayscale if necessary, then resize to 256x256 pixels.
    3. Normalize pixel values from [0, 255] to [-1.0, 1.0].
    4. Convert to a float32 PyTorch tensor and unsqueeze to [1, 1, 256, 256] shape.

    Args:
        pil_image (Image.Image): Input PIL Image.

    Returns:
        torch.Tensor: Preprocessed tensor of shape [1, 1, 256, 256] (dtype=torch.float32).
    """
    # Convert PIL Image to numpy array
    img_np = np.array(pil_image)

    # Convert to grayscale based on the input channel count
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_np.shape[2] == 4:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        else:
            gray = img_np
    else:
        gray = img_np

    # Resize to exactly 256x256
    gray_resized = cv2.resize(gray, (256, 256))

    # Normalize to [-1.0, 1.0]
    gray_normalized = (gray_resized.astype(np.float32) / 127.5) - 1.0

    # Convert to torch tensor and add batch and channel dimensions: [1, 1, 256, 256]
    tensor = torch.from_numpy(gray_normalized).unsqueeze(0).unsqueeze(0)

    return tensor


def run_inference(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """
    Execute model forward pass to translate grayscale IR tensor into Optical RGB image.

    Execution steps:
    1. Detect if a CUDA GPU is available, else default to CPU.
    2. Move both model and input tensor to the target device.
    3. Execute the forward pass within a `torch.no_grad()` context.
    4. Move the output tensor back to CPU, squeeze batch dimension, permute to
       channel-last format [256, 256, 3], and convert to numpy.
    5. Denormalize pixel intensities from [-1.0, 1.0] back to [0, 255] uint8.

    Args:
        model (nn.Module): Instantiated DualStreamIR2RGB PyTorch model.
        tensor (torch.Tensor): Preprocessed input tensor of shape [1, 1, 256, 256].

    Returns:
        np.ndarray: Translated optical image array of shape [256, 256, 3] (dtype=np.uint8).
    """
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model and input tensor to device
    model.to(device)
    tensor = tensor.to(device)

    # Wrap forward pass in no_grad to disable gradient computation and save memory
    with torch.no_grad():
        output = model(tensor)

    # Move output to CPU, squeeze batch dimension, and permute to [H, W, C] format
    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Denormalize output from [-1.0, 1.0] to [0, 255]
    denormalized = ((output_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    return denormalized
