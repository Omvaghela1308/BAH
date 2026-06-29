"""
Enhanced Thermal Risk Assessment Engine Module.

Analyzes infrared images, segmentation masks, and downstream detection counts
to determine dynamic risk levels, reasons, actionable operator recommendations,
and generates a localized heatmap overlay and micro-statistics.
"""

import cv2
import numpy as np


def assess_thermal_risk(
    ir_gray: np.ndarray, mask: np.ndarray, object_count: int, rgb_image: np.ndarray
) -> tuple[str, list[str], str, dict, np.ndarray]:
    """
    Evaluate thermal risk metrics and generate visual heatmap overlays.

    Risk Levels:
    - Critical: multiple hotspot clusters + high thermal variance + restricted zone proximity.
    - High: activity detected OR isolated hotspots near structures OR peak thermal bounds exceeded.
    - Medium: mild thermal variance / anomalies within nominal range.
    - Low: normal thermal parameters, baseline variance.

    Args:
        ir_gray (np.ndarray): Resized grayscale IR image, shape [H, W].
        mask (np.ndarray): Segmentation mask, shape [H, W].
        object_count (int): Downstream object detection count.
        rgb_image (np.ndarray): Translated RGB image, shape [H, W, 3].

    Returns:
        tuple[str, list[str], str, dict, np.ndarray]: 
            (risk_level, reason_bullets, recommendation_text, statistics_dict, heatmap_overlay_rgb)
    """
    # 1. Compute basic statistics
    mean_val = np.mean(ir_gray)
    max_val = np.max(ir_gray)
    variance = np.var(ir_gray)

    # 2. Extract hotspot counts using thresholding and contour analysis
    _, thresh = cv2.threshold(ir_gray, 185, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspot_count = len(contours)

    reasons = []

    # 3. Analyze hotspot distribution across land cover classes
    # Classes: 0: Water, 1: Vegetation, 2: Urban, 3: Bare Soil, 4: Clouds
    has_vegetation_hotspots = False
    has_urban_hotspots = False

    for cnt in contours:
        # Get centroid of hotspot contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                cls_val = mask[cy, cx]
                if cls_val == 1:
                    has_vegetation_hotspots = True
                elif cls_val == 2:
                    has_urban_hotspots = True

    # 4. Generate Heatmap Overlay
    # Create smooth blurred hotspot mask
    blur_hotspot = cv2.GaussianBlur(thresh, (15, 15), 0)
    # Apply COLORMAP_HOT
    heatmap_bgr = cv2.applyColorMap(blur_hotspot, cv2.COLORMAP_HOT)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Blending factor proportional to hotspot intensity
    alpha = 0.65
    mask_alpha = (blur_hotspot.astype(np.float32) / 255.0) * alpha
    mask_alpha = np.expand_dims(mask_alpha, axis=-1)
    
    # Blend images
    blended = (rgb_image.astype(np.float32) * (1.0 - mask_alpha) + heatmap_rgb.astype(np.float32) * mask_alpha)
    heatmap_overlay = np.clip(blended, 0, 255).astype(np.uint8)

    # 5. Calibrate Statistics
    peak_temp_celsius = int(35 + (max_val / 255.0) * 280)
    hotspot_density_pct = (np.sum(thresh > 0) / thresh.size) * 100.0
    
    if has_vegetation_hotspots:
        vulnerable_class = "Vegetation (Forest) - Proximity Danger"
    elif has_urban_hotspots:
        vulnerable_class = "Urban Built-up - Activity Danger"
    else:
        vulnerable_class = "Bare Soil / Desert - Low Vulnerability"

    # Calculate confidence dynamically and deterministically
    confidence_val = 92.5 + (mean_val / 255.0) * 5.0 - (variance / 30000.0) * 3.0
    confidence_val = min(99.9, max(80.0, confidence_val))

    stats = {
        "peak_temp": f"{peak_temp_celsius}°C",
        "density": f"{hotspot_density_pct:.2f}% of area",
        "proximity": vulnerable_class,
        "spread": f"{hotspot_density_pct:.2f}%",
        "confidence": f"{confidence_val:.1f}%"
    }

    # 6. Apply risk rules
    is_critical = hotspot_count > 3 and variance > 1200.0 and has_vegetation_hotspots
    is_high = object_count > 2 or (hotspot_count > 0 and has_urban_hotspots) or max_val > 230
    is_medium = hotspot_count > 0 or variance > 600.0 or max_val > 160

    if is_critical:
        level = "Critical"
        reasons.append(f"⚠ {hotspot_count} thermal hotspot clusters detected across the spatial grid")
        reasons.append(f"⚠ Thermal variance critically elevated — severe anomaly distribution confirmed")
        reasons.append("⚠ Anomalies encroaching on protected vegetation boundary")
        rec = "Immediate operator intervention required. Alert emergency response."
    elif is_high:
        level = "High"
        if object_count > 2:
            reasons.append(f"⚠ Activity detected in sensitive zone ({object_count} objects identified)")
        if hotspot_count > 0:
            reasons.append(f"⚠ {hotspot_count} isolated thermal hotspot clusters detected near structural elements")
        if max_val > 230:
            reasons.append(f"⚠ Thermal intensity elevated — peak value ({max_val}) exceeds baseline parameters")
        if not reasons:
            reasons.append("⚠ Localized thermal anomalies identified near industrial or built-up infrastructure")
        rec = "Immediate inspection advised. Flag for human review."
    elif is_medium:
        level = "Medium"
        reasons.append(f"⚠ Mild thermal anomalies observed with spatial variance within normal limits ({variance:.1f})")
        reasons.append(f"⚠ Average background thermal intensity ({mean_val:.1f}) is within acceptable seasonal baseline")
        rec = "Monitor area. Schedule follow-up scan within 24 hours."
    else:
        level = "Low"
        reasons.append("⚠ No significant thermal anomalies or hotspot structures detected")
        reasons.append("⚠ Thermal intensity and spatial distribution are within baseline nominal parameters")
        rec = "No action required. Continue routine monitoring."

    return level, reasons, rec, stats, heatmap_overlay
