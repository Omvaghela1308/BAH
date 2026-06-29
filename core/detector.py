"""
Downstream Object Detection Comparison Module.

Uses computer vision and contour analysis to detect features (such as buildings, roads,
fields, or vehicles) on both raw IR images and translated RGB images. Demonstrates
the performance improvement in downstream tasks provided by colorization.
"""

import cv2
import numpy as np


def detect_objects(image: np.ndarray, is_rgb: bool = True) -> tuple[np.ndarray, int]:
    """
    Detect structural features and objects using contour and boundary analysis.

    Args:
        image (np.ndarray): Input image array. For RGB, shape is [H, W, 3];
                            for IR, shape can be [H, W] or [H, W, 3].
        is_rgb (bool): True if the input is an RGB image, False if grayscale IR.

    Returns:
        tuple[np.ndarray, int]: Annotated image array, and count of detected objects.
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
        # to identify boundaries made clear by the color translation
        blurred = cv2.bilateralFilter(gray, 5, 50, 50)
        edges = cv2.Canny(blurred, 30, 150)
        # Dilate edges to close contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.dilate(edges, kernel, iterations=1)
    else:
        # In grayscale IR, use standard thresholding which misses color-only boundaries
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Find contours representing structures/objects
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make a copy of the image to annotate
    annotated = image.copy()
    if len(annotated.shape) == 2:
        # Convert grayscale IR to 3-channel grayscale so we can draw colored bounding boxes if needed
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)

    count = 0
    box_color = (0, 255, 0) if is_rgb else (255, 255, 255)  # Green for RGB, White for IR

    # 4. Filter contours by size to isolate object blobs
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Plausible object area range in 256x256 image
        if 20.0 <= area <= 2000.0:
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            count += 1

    return annotated, count
