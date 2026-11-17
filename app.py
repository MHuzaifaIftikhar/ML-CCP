"""
=============================================================================
LAB 13 INTEGRATION: REAL-TIME ASL GESTURE RECOGNITION & AR UI OVERLAY
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
Dual-Dataset Live Webcam Translator:
  - Supports switching between KAGGLE BENCHMARK (clean) and CUSTOM LAB (webcam) models
  - Video stream acquisition via OpenCV
  - Real-time 21 3D hand landmark extraction via universal HandTracker
  - Wrist-relative coordinate translation and invariant scaling (Lab 01)
  - Standardization using dataset-specific StandardScaler
  - Multi-class probability inference using SVM (Lab 06) or KNN (Lab 08)
  - Temporal Frame-Smoothing Engine using collections.deque & Counter (Lab 13 / 14)
  - Augmented Reality (AR) HUD visual overlay showing active translated sign,
    dataset mode, confidence score, FPS, and skeletal hand connections.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
import joblib

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.preprocessing import normalize_live_landmarks
from src.utils import (
    TemporalSmoother,
    FPSCounter,
    draw_hud_overlay
)
from src.hand_tracker import HandTracker


# =============================================================================
# 1. DUAL-DATASET MODEL AND ARTIFACT LOADER
# =============================================================================
def load_all_artifacts() -> Dict[str, Dict[str, Any]]:
    """
    Loads model artifacts for all 3 datasets:
      1. 'kaggle': Clean Kaggle Benchmark models (A-Z)
      2. 'custom': Custom Lab Dataset models (A-Z)
      3. 'asl36': ASL 36 Classes Dataset models (0-9, A-Z)
    Each bundle contains SVM, KNN, Random Forest, and StandardScaler.
    """
    artifacts = {}

    # Check and train if missing
    if (not config.KAGGLE_SVM_PATH.exists() or 
        not config.CUSTOM_SVM_PATH.exists() or 
        not config.ASL36_SVM_PATH.exists()):
        print("[!] Multi-dataset models not found in models/ directory.")
        print("[*] Initiating automated training pipeline for all datasets...")
        from src.train import run_cross_dataset_benchmarks
        run_cross_dataset_benchmarks()

    # Load Kaggle artifacts
    try:
        artifacts["kaggle"] = {
            "name": "Kaggle Benchmark (A-Z)",
            "svm": joblib.load(config.KAGGLE_SVM_PATH),
            "knn": joblib.load(config.KAGGLE_KNN_PATH),
            "rf": joblib.load(config.KAGGLE_RF_PATH) if config.KAGGLE_RF_PATH.exists() else None,
            "scaler": joblib.load(config.KAGGLE_SCALER_PATH),
            "classes": 26
        }
        print(f"[+] Loaded Kaggle Benchmark artifacts from: {config.KAGGLE_MODELS_DIR}")
    except Exception as e:
        print(f"[!] Warning loading Kaggle artifacts: {e}")

    # Load Custom Lab artifacts
    try:
        artifacts["custom"] = {
            "name": "Custom Lab Data (A-Z)",
            "svm": joblib.load(config.CUSTOM_SVM_PATH),
            "knn": joblib.load(config.CUSTOM_KNN_PATH),
            "rf": joblib.load(config.CUSTOM_RF_PATH) if config.CUSTOM_RF_PATH.exists() else None,
            "scaler": joblib.load(config.CUSTOM_SCALER_PATH),
            "classes": 26
        }
        print(f"[+] Loaded Custom Lab artifacts from: {config.CUSTOM_MODELS_DIR}")
    except Exception as e:
        print(f"[!] Warning loading Custom artifacts: {e}")

    # Load ASL 36-Classes artifacts
    try:
        artifacts["asl36"] = {
            "name": "ASL 36 Classes (0-9, A-Z)",
            "svm": joblib.load(config.ASL36_SVM_PATH),
            "knn": joblib.load(config.ASL36_KNN_PATH),
            "rf": joblib.load(config.ASL36_RF_PATH) if config.ASL36_RF_PATH.exists() else None,
            "scaler": joblib.load(config.ASL36_SCALER_PATH),
            "classes": 36
        }
        print(f"[+] Loaded ASL 36-Classes artifacts from: {config.ASL36_MODELS_DIR}")
    except Exception as e:
        print(f"[!] Warning loading ASL 36-Classes artifacts: {e}")

    # Fallback to top-level models if needed
    if "kaggle" not in artifacts and config.SVM_MODEL_PATH.exists():
        artifacts["kaggle"] = {
            "name": "Default Model (A-Z)",
            "svm": joblib.load(config.SVM_MODEL_PATH),
            "knn": joblib.load(config.KNN_MODEL_PATH),
            "rf": joblib.load(config.RF_MODEL_PATH) if config.RF_MODEL_PATH.exists() else None,
            "scaler": joblib.load(config.SCALER_PATH),
            "classes": 26
        }

    return artifacts


# =============================================================================
# 2. WEBCAM INITIALIZATION WITH MULTI-INDEX FALLBACK
# =============================================================================
def initialize_webcam(preferred_index: int = config.CAMERA_INDEX) -> cv2.VideoCapture:
    """Initializes OpenCV VideoCapture with fallback support."""
    print(f"[*] Opening webcam device index {preferred_index}...")
    cap = cv2.VideoCapture(preferred_index)

    if not cap.isOpened():
        for fallback_idx in [1, 2, 0]:
            if fallback_idx != preferred_index:
                print(f"[!] Primary camera failed. Attempting fallback index {fallback_idx}...")
                cap = cv2.VideoCapture(fallback_idx)
                if cap.isOpened():
                    print(f"[+] Successfully opened webcam on index {fallback_idx}")
                    break

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open any webcam video stream. Please verify that a camera is connected "
            "and that no other application is using it."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
    return cap


# =============================================================================
# 3. REAL-TIME TRANSLATOR ENGINE (Lab 13 Main Loop)
# =============================================================================
def run_realtime_translator():
    """
    Main execution loop for real-time ASL sign translation with dual-dataset support.
    """
    print("\n" + "=" * 76)
    print("STARTING DUAL-DATASET ASL SIGN TRANSLATOR (ML x AR)")
    print("DUET AI-3202 Machine Learning Practical Manual - Lab 13 & Lab 14 CCA")
    print("=" * 76)

    # 1. Load All Dataset Artifacts
    all_artifacts = load_all_artifacts()
    available_datasets = [d for d in ["kaggle", "custom", "asl36"] if d in all_artifacts]
    if not available_datasets:
        available_datasets = list(all_artifacts.keys())

    # Default to Kaggle (clean benchmark) for optimal accuracy out of the box
    current_dataset = "kaggle" if "kaggle" in available_datasets else available_datasets[0]
    available_models = ["svm", "knn", "rf"]
    current_model_type = "svm"  # 'svm', 'knn', or 'rf'

    # 2. Initialize Universal Hand Tracker
    hand_tracker = HandTracker()

    # 3. Initialize Temporal Smoother and FPS Counter
    smoother = TemporalSmoother(
        window_size=config.SMOOTHING_WINDOW_SIZE,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        min_stable_count=config.CONSECUTIVE_STABLE_FRAMES
    )
    fps_counter = FPSCounter()
    temporal_smoothing_enabled = True

    # 4. Open Webcam Feed
    cap = initialize_webcam()
    window_name = "Multi-Dataset ASL Sign Language Translator (DUET AI-3202)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    print("\n[INFO] Interactive Controls:")
    print("   * Press 'q' to Exit application.")
    print("   * Press 'd' to Cycle Dataset Mode: [KAGGLE] -> [CUSTOM] -> [ASL-36 (0-9, A-Z)].")
    print("   * Press 'm' to Cycle Classifier: [SVM (RBF)] -> [KNN (k=5)] -> [RANDOM FOREST].")
    print("   * Press 's' to Toggle Temporal Frame Smoothing (ON / OFF).")
    print("   * Press 'c' to Clear temporal smoothing buffer.\n")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Active model and scaler based on current selection
            active_bundle = all_artifacts[current_dataset]
            active_model = active_bundle.get(current_model_type) or active_bundle.get("svm")
            active_scaler = active_bundle["scaler"]
            is_36_class = (active_bundle.get("classes", 26) == 36 or current_dataset == "asl36")

            # Process hand landmarks
            landmarks = hand_tracker.process(frame)

            hand_detected = False
            raw_letter: Optional[str] = None
            confidence = 0.0

            if landmarks is not None:
                hand_detected = True
                hand_tracker.draw_landmarks(frame, landmarks)

                try:
                    # Normalize landmarks relative to wrist using active dataset's scaler
                    scaled_features = normalize_live_landmarks(landmarks, scaler=active_scaler)

                    # Multi-class probability prediction
                    if hasattr(active_model, "predict_proba"):
                        proba_dist = active_model.predict_proba(scaled_features)[0]
                        pred_idx = int(np.argmax(proba_dist))
                        confidence = float(proba_dist[pred_idx])
                    else:
                        pred_idx = int(active_model.predict(scaled_features)[0])
                        confidence = 1.0

                    # Map index to class label (26 letters vs 36 classes)
                    if is_36_class:
                        raw_letter = config.IDX_TO_LABEL_36.get(pred_idx, "?")
                    else:
                        raw_letter = config.IDX_TO_LABEL.get(pred_idx, "?")

                except Exception as e:
                    print(f"[!] Inference error: {e}")

            # Update temporal smoothing queue
            if hand_detected and raw_letter:
                if temporal_smoothing_enabled:
                    active_letter, smooth_conf, is_stable = smoother.update(raw_letter, confidence)
                else:
                    active_letter = raw_letter
                    smooth_conf = confidence
                    is_stable = (confidence >= config.CONFIDENCE_THRESHOLD)
            else:
                active_letter, smooth_conf, is_stable = smoother.update("?", 0.0)
                is_stable = False

            fps = fps_counter.update()

            # Construct dynamic status badge: e.g. "[ASL-36] RANDOM FOREST"
            if current_dataset == "asl36":
                dataset_tag = "ASL-36"
            elif current_dataset == "kaggle":
                dataset_tag = "KAGGLE"
            else:
                dataset_tag = "CUSTOM"

            if current_model_type == "svm":
                model_tag = "SVM (RBF)"
            elif current_model_type == "knn":
                model_tag = "KNN (k=5)"
            elif current_model_type == "rf":
                model_tag = "RANDOM FOREST"
            else:
                model_tag = current_model_type.upper()

            smooth_tag = "" if temporal_smoothing_enabled else " (Raw)"
            display_model_name = f"[{dataset_tag}] {model_tag}{smooth_tag}"

            # Render Augmented Reality HUD overlay
            frame = draw_hud_overlay(
                frame=frame,
                active_letter=active_letter,
                confidence=smooth_conf,
                is_stable=is_stable,
                fps=fps,
                recent_history=smoother.get_recent_history(),
                model_name=display_model_name,
                hand_detected=hand_detected
            )

            cv2.imshow(window_name, frame)

            # Keyboard event handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[*] Exit signal received. Shutting down...")
                break
            elif key == ord('d'):  # Cycle Dataset Mode (Kaggle -> Custom -> ASL-36)
                if len(available_datasets) > 1:
                    curr_idx = available_datasets.index(current_dataset)
                    current_dataset = available_datasets[(curr_idx + 1) % len(available_datasets)]
                    smoother.clear()
                    print(f"[*] Dataset Mode toggled to: {all_artifacts[current_dataset]['name'].upper()}")
            elif key == ord('m'):  # Cycle Classifier (SVM -> KNN -> Random Forest)
                curr_model_idx = available_models.index(current_model_type)
                current_model_type = available_models[(curr_model_idx + 1) % len(available_models)]
                smoother.clear()
                print(f"[*] Classifier toggled to: {current_model_type.upper()}")
                smoother.clear()
                print(f"[*] Classifier toggled to: {current_model_type.upper()}")
            elif key == ord('s'):  # Toggle Smoothing
                temporal_smoothing_enabled = not temporal_smoothing_enabled
                state_str = "ENABLED" if temporal_smoothing_enabled else "DISABLED"
                print(f"[*] Temporal Frame-Smoothing {state_str}.")
            elif key == ord('c'):  # Clear Smoothing Buffer
                smoother.clear()
                print("[*] Temporal smoothing buffer cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.close()
        print("[+] Camera stream and HandTracker released cleanly.")
        print("[+] Application closed successfully.")


if __name__ == "__main__":
    run_realtime_translator()
