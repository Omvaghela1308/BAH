"""
Semantic Land Cover Segmentation Module.

Provides rule-based per-pixel land cover classification and visualization functions
to guide the colorization process and satisfy semantic consistency objectives.
"""

import cv2
import numpy as np


def segment_land_cover(ir_gray: np.ndarray, rgb_image: np.ndarray) -> np.ndarray:
    """
    Perform semantic land-cover segmentation using a per-pixel classification heuristic.

    Classes:
    - 0: Water (low IR intensity, ir_val < 45)
    - 1: Vegetation (ir_val between 135 and 195, natural muted green)
    - 2: Urban (ir_val between 45 and 90, greyish structures)
    - 3: Bare Soil / Desert (ir_val between 90 and 135, tan/beige colors)
    - 4: Storm Systems (ir_val between 195 and 225, grey-white atmospheric gradient)
    - 5: Clouds / Snow (ir_val >= 225, bright white/light grey)

    Args:
        ir_gray (np.ndarray): Grayscale input Infrared image, shape [H, W].
        rgb_image (np.ndarray): Generated/translated RGB image, shape [H, W, 3].

    Returns:
        np.ndarray: Integer segmentation mask of shape [H, W], values 0 to 5.
    """
    h, w = ir_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            ir_val = ir_gray[y, x]

            if ir_val < 45:
                mask[y, x] = 0  # Water
            elif ir_val < 90:
                mask[y, x] = 2  # Urban
            elif ir_val < 135:
                mask[y, x] = 3  # Bare Soil
            elif ir_val < 195:
                mask[y, x] = 1  # Vegetation
            elif ir_val < 225:
                mask[y, x] = 4  # Storm Systems
            else:
                mask[y, x] = 5  # Clouds / Snow

    return mask


def color_code_mask(mask: np.ndarray) -> np.ndarray:
    """
    Color-code the segmentation mask for visual verification.

    Colors:
    - Water (0) -> Blue [29, 92, 150]
    - Vegetation (1) -> Muted Green [56, 102, 65]
    - Urban (2) -> Grey [116, 140, 171]
    - Bare Soil (3) -> Tan [212, 163, 115]
    - Storm Systems (4) -> Atmospheric Grey [165, 175, 185]
    - Cloud / Snow (5) -> White [248, 249, 250]

    Args:
        mask (np.ndarray): Integer segmentation mask of shape [H, W].

    Returns:
        np.ndarray: Color-coded RGB mask of shape [H, W, 3], dtype uint8.
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # Class colors
    colors = {
        0: [29, 92, 150],     # Water (Blue)
        1: [56, 102, 65],     # Vegetation (Muted Green)
        2: [116, 140, 171],   # Urban (Grey)
        3: [212, 163, 115],   # Bare Soil (Tan)
        4: [165, 175, 185],   # Storm Systems (Atmospheric Grey)
        5: [248, 249, 250]    # Cloud/Snow (White)
    }

    for val, color in colors.items():
        class_indices = (mask == val)
        colored[class_indices] = color

    return colored
