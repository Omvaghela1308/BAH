"""
Downstream Object Detection Comparison Module.

Uses computer vision and contour analysis to detect features (such as buildings, roads,
fields, or vehicles) on both raw IR images and translated RGB images. Demonstrates
the performance improvement in downstream tasks provided by colorization.
"""

import cv2
import numpy as np


def detect_objects(image: np.ndarray, is_rgb: bool = True, mask: np.ndarray = None, return_details: bool = False) -> tuple:
    """
    Detect structural features and objects using contour and boundary analysis.

    Args:
        image (np.ndarray): Input image array.
        is_rgb (bool): True if the input is an RGB image, False if grayscale IR.
        mask (np.ndarray): Semantic segmentation mask to identify categories.
        return_details (bool): Whether to return a breakdown dictionary of classified objects.

    Returns:
        tuple: (annotated_image, count) or (annotated_image, count, details_dict)
    """
    # 1. Convert to grayscale and normalize
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    # 2. Extract structures based on channel type
    if is_rgb:
        # In colorized RGB, use edge-preserving smoothing and adaptive thresholding
        blurred = cv2.bilateralFilter(gray, 5, 50, 50)
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.dilate(edges, kernel, iterations=1)
    else:
        # In grayscale IR, use standard thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Find contours representing structures/objects
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make a copy of the image to annotate
    annotated = image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)

    count = 0
    box_color = (0, 255, 0) if is_rgb else (255, 255, 255)

    # Categories tracking
    details = {
        "Building (House)": 0,
        "Tree / Vegetation": 0,
        "Water Body": 0,
        "Vehicle / Road / Ground": 0,
    }

    # 4. Filter contours by size to isolate object blobs
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20.0 <= area <= 2000.0:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            count += 1

            # Classify object category based on mask value at centroid
            if mask is not None:
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                    cls_val = mask[cy, cx]
                    if cls_val == 0:
                        details["Water Body"] += 1
                    elif cls_val == 1:
                        details["Tree / Vegetation"] += 1
                    elif cls_val == 2:
                        details["Building (House)"] += 1
                    else:
                        details["Vehicle / Road / Ground"] += 1
                else:
                    details["Vehicle / Road / Ground"] += 1
            else:
                # Default heuristic if mask is unavailable
                aspect_ratio = float(w) / max(1, h)
                if 0.9 <= aspect_ratio <= 1.1:
                    details["Building (House)"] += 1
                elif area < 100:
                    details["Vehicle / Road / Ground"] += 1
                else:
                    details["Tree / Vegetation"] += 1

    if return_details:
        return annotated, count, details
    return annotated, count
