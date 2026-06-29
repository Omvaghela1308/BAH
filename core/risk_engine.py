"""
AI Thermal Risk Assessment Engine.

Provides modular analysis and scoring functions to evaluate thermal anomalies,
hotspot distribution, background brightness, structural variance, and confidence
to estimate overall risk levels and provide context-aware operator recommendations.
"""

import cv2
import numpy as np


def extract_features(ir_image: np.ndarray, rgb_image: np.ndarray) -> dict:
    """
    Extract dynamic features from the input Infrared and output RGB images.

    Args:
        ir_image (np.ndarray): Grayscale or RGB infrared input image.
        rgb_image (np.ndarray): Translated optical RGB output image.

    Returns:
        dict: Extracted metrics including brightness, thermal_intensity,
              hotspot_count, objects_detected, anomaly_score, and confidence.
    """
    # Ensure grayscale for input IR image
    if len(ir_image.shape) == 3:
        if ir_image.shape[2] == 3:
            ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_RGB2GRAY)
        elif ir_image.shape[2] == 4:
            ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_RGBA2GRAY)
        else:
            ir_gray = ir_image[:, :, 0]
    else:
        ir_gray = ir_image

    # Ensure 256x256 resolution for consistent feature scale
    if ir_gray.shape != (256, 256):
        ir_gray = cv2.resize(ir_gray, (256, 256))

    # 1. Background Brightness (average pixel intensity, scaled to 0-100)
    mean_val = np.mean(ir_gray)
    brightness = float((mean_val / 255.0) * 100.0)

    # 2. Thermal Intensity (99.5th percentile to capture peak heat zones, scaled to 0-100)
    p99_val = np.percentile(ir_gray, 99.5)
    thermal_intensity = float((p99_val / 255.0) * 100.0)

    # 3. Hotspot Count (count high-contrast thermal areas using thresholding and contours)
    # Threshold at high intensity (e.g. 200 out of 255)
    _, thresh = cv2.threshold(ir_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspot_count = len(contours)

    # 4. Objects Detected (heuristic: identify isolated blobs with plausible area sizes)
    # Area must be between 15 and 1500 pixels in a 256x256 resolution frame
    objects_detected = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 15.0 <= area <= 1500.0:
            objects_detected += 1

    # 5. Anomaly Score (intensity variance represented by standard deviation, scaled to 0-100)
    std_val = np.std(ir_gray)
    anomaly_score = float(min(100.0, (std_val / 64.0) * 100.0))

    # 6. AI Confidence (estimate based on contrast quality and signature clarity)
    if hotspot_count > 0:
        base_conf = 80.0 + min(19.0, objects_detected * 3.0)
    else:
        base_conf = 95.0 - min(15.0, anomaly_score * 0.1)
    confidence = float(np.clip(base_conf, 50.0, 99.0))

    return {
        "brightness": brightness,
        "thermal_intensity": thermal_intensity,
        "hotspot_count": hotspot_count,
        "objects_detected": objects_detected,
        "anomaly_score": anomaly_score,
        "confidence": confidence,
    }


def default_risk_formula(features: dict) -> dict:
    """
    Default scoring formula combining thermal metrics using weighted fusion.

    Args:
        features (dict): Extracted features dictionary.

    Returns:
        dict: Calculated score (int) and threat level (str).
    """
    # Weighted contribution
    w_thermal = 0.40  # Peak heat signature weight
    w_hotspots = 0.20  # Frequency of anomaly points
    w_objects = 0.20  # Isolated threat structure weight
    w_anomaly = 0.20  # Spatial deviation weight

    thermal_score = features["thermal_intensity"]
    # Normalize hotspot count: 5+ hotspots corresponds to maximum danger contribution
    hotspot_score = min(100.0, (features["hotspot_count"] / 5.0) * 100.0)
    # Normalize objects: 3+ structures corresponds to maximum threat density
    objects_score = min(100.0, (features["objects_detected"] / 3.0) * 100.0)
    anomaly_score = features["anomaly_score"]

    raw_score = (
        thermal_score * w_thermal +
        hotspot_score * w_hotspots +
        objects_score * w_objects +
        anomaly_score * w_anomaly
    )
    score = int(np.clip(raw_score, 0, 100))

    # Threat Level Classification
    if score <= 25:
        level = "LOW"
    elif score <= 50:
        level = "MEDIUM"
    elif score <= 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return {
        "score": score,
        "level": level,
    }


def calculate_risk(features: dict, formula_fn=None) -> dict:
    """
    Modular risk calculator accepting custom formula functions for future expansions.

    Args:
        features (dict): Extracted features dictionary.
        formula_fn (callable, optional): Custom formula function. Defaults to default_risk_formula.

    Returns:
        dict: Calculation results containing "score" and "level".
    """
    if formula_fn is None:
        formula_fn = default_risk_formula
    return formula_fn(features)


def generate_analysis(features: dict, score: int, level: str) -> list[str]:
    """
    Dynamically construct descriptive bullet points for the AI Analysis section.

    Args:
        features (dict): Extracted features.
        score (int): Risk score.
        level (str): Threat level.

    Returns:
        list[str]: Explanations of the current thermal profile.
    """
    analysis = []

    # Heat evaluation
    if features["thermal_intensity"] > 75.0:
        analysis.append("Critical heat signature detected in localized pixels.")
    elif features["thermal_intensity"] > 50.0:
        analysis.append("High thermal concentration observed in focal areas.")
    else:
        analysis.append("Thermal signature remains within expected baseline range.")

    # Hotspot count evaluation
    if features["hotspot_count"] > 3:
        analysis.append(f"Multiple hotspot clusters ({features['hotspot_count']}) detected in spatial grid.")
    elif features["hotspot_count"] > 0:
        analysis.append("Isolated hotspot region identified, indicating point anomaly.")

    # Distinct structures
    if features["objects_detected"] > 0:
        label = "structure" if features["objects_detected"] == 1 else "structures"
        analysis.append(f"Detected {features['objects_detected']} distinct heat signature {label}.")

    # Anomaly profile
    if features["anomaly_score"] > 60.0:
        analysis.append("Thermal spatial deviation exceeds nominal variance levels.")

    # Level-specific contextual markers
    if level == "CRITICAL":
        analysis.append("High-risk environment confirmed. Operator intervention required.")
    elif level == "HIGH" and features["objects_detected"] > 0:
        analysis.append("Unidentified structural thermal signatures warrant caution.")

    return analysis


def generate_recommendation(score: int, level: str, features: dict) -> str:
    """
    Construct an intelligent recommendation message based on risk level and features.

    Args:
        score (int): Overall score.
        level (str): Threat level.
        features (dict): Extracted features.

    Returns:
        str: Actionable recommendation.
    """
    if level == "LOW":
        return "Continue monitoring. Thermal signatures match nominal baseline patterns."
    elif level == "MEDIUM":
        return "Increase surveillance frequency. Monitor the area for further heat increases or shape progression."
    elif level == "HIGH":
        if features["objects_detected"] > 0:
            return "Potential intrusion detected. Verify the area manually using ground units."
        return "High thermal anomaly observed. Double surveillance frequency and review auxiliary sensors."
    else:  # CRITICAL
        if features["thermal_intensity"] > 80.0:
            return "Extreme heat detected. Immediate operator attention recommended to address potential fire or safety hazard."
        return "Immediate operator attention and ground verification recommended. Potential security breach or hazard."


def classify_event(score: int) -> dict:
    """
    Classify the current event based on risk score.

    Args:
        score (int): Overall score.

    Returns:
        dict: Name, color (Hex), and visual icon of classification.
    """
    if score <= 25:
        return {"name": "Normal", "color": "#39d353", "icon": "🟢"}
    elif score <= 50:
        return {"name": "Suspicious", "color": "#f1e05a", "icon": "🟡"}
    elif score <= 75:
        return {"name": "Critical", "color": "#f0883e", "icon": "🟠"}
    else:
        return {"name": "Emergency", "color": "#f85149", "icon": "🔴"}


def get_risk_factors(features: dict) -> list[str]:
    """
    Determine active risk factors based on feature thresholds.

    Args:
        features (dict): Extracted features.

    Returns:
        list[str]: Detected risk factor labels.
    """
    factors = []
    # 1. Human Presence: Heat signature objects detected
    if features["objects_detected"] > 0:
        factors.append("Human Presence")
    # 2. High Heat: Normalized thermal intensity above 50%
    if features["thermal_intensity"] > 55.0:
        factors.append("High Heat")
    # 3. Thermal Cluster: Multiple hotspots
    if features["hotspot_count"] >= 3:
        factors.append("Thermal Cluster")
    # 4. Night Activity: Very low overall brightness (indicative of night/dark scene)
    if features["brightness"] < 35.0:
        factors.append("Night Activity")
    # 5. Unknown Object: Structural objects with high overall anomaly
    if features["objects_detected"] > 2 or features["anomaly_score"] > 70.0:
        factors.append("Unknown Object")
    # 6. Fire Risk: Very high localized heat + hotspots
    if features["thermal_intensity"] > 80.0 and features["hotspot_count"] > 0:
        factors.append("Fire Risk")
    return factors
