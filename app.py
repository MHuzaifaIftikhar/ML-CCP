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
    draw_hud_overlay,
    show_evaluation_graphs_viewer,
    is_thumbs_up
)
from src.hand_tracker import HandTracker
from src.sentence_engine import SentenceSuggestionEngine, SpeechEngine


# =============================================================================
# 1. DUAL-DATASET MODEL AND ARTIFACT LOADER
# =============================================================================
def load_all_artifacts() -> Dict[str, Dict[str, Any]]:
    """
    Loads model artifacts for both:
      1. 'kaggle': Clean Kaggle Benchmark models
      2. 'custom': Custom Lab Dataset models
    """
    artifacts = {}

    # Check and train if missing
    if not config.KAGGLE_SVM_PATH.exists() or not config.CUSTOM_SVM_PATH.exists():
        print("[!] Multi-dataset models not found in models/ directory.")
        print("[*] Initiating automated training pipeline for all datasets...")
        from src.train import run_cross_dataset_benchmarks
        run_cross_dataset_benchmarks()

    # Load Kaggle artifacts
    try:
        artifacts["kaggle"] = {
            "name": "Kaggle Benchmark",
            "svm": joblib.load(config.KAGGLE_SVM_PATH),
            "knn": joblib.load(config.KAGGLE_KNN_PATH),
            "scaler": joblib.load(config.KAGGLE_SCALER_PATH)
        }
        print(f"[+] Loaded Kaggle Benchmark artifacts from: {config.KAGGLE_MODELS_DIR}")
    except Exception as e:
        print(f"[!] Warning loading Kaggle artifacts: {e}")

    # Load Custom Lab artifacts
    try:
        artifacts["custom"] = {
            "name": "Custom Lab Data",
            "svm": joblib.load(config.CUSTOM_SVM_PATH),
            "knn": joblib.load(config.CUSTOM_KNN_PATH),
            "scaler": joblib.load(config.CUSTOM_SCALER_PATH)
        }
        print(f"[+] Loaded Custom Lab artifacts from: {config.CUSTOM_MODELS_DIR}")
    except Exception as e:
        print(f"[!] Warning loading Custom artifacts: {e}")

    # Fallback to top-level models if either is missing
    if "kaggle" not in artifacts and config.SVM_MODEL_PATH.exists():
        artifacts["kaggle"] = {
            "name": "Default Model",
            "svm": joblib.load(config.SVM_MODEL_PATH),
            "knn": joblib.load(config.KNN_MODEL_PATH),
            "scaler": joblib.load(config.SCALER_PATH)
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
    available_datasets = list(all_artifacts.keys())
    
    # Default to Custom Lab dataset (your recorded webcam data)
    current_dataset = "custom" if "custom" in available_datasets else available_datasets[0]
    current_model_type = "svm"  # 'svm' or 'knn'

    # 2. Initialize Universal Hand Tracker
    hand_tracker = HandTracker()

    # 3. Initialize Temporal Smoother, FPS Counter, Sentence & Speech Engines
    smoother = TemporalSmoother(
        window_size=config.SMOOTHING_WINDOW_SIZE,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        min_stable_count=config.CONSECUTIVE_STABLE_FRAMES
    )
    fps_counter = FPSCounter()
    temporal_smoothing_enabled = True

    sentence_engine = SentenceSuggestionEngine()
    speech_engine = SpeechEngine()
    auto_type_enabled = True
    dwell_duration = 1.2       # Seconds to hold a stable sign before auto-adding
    dwell_cooldown = 1.5       # Seconds before same sign can auto-add again
    dwell_letter: Optional[str] = None
    dwell_start_time: float = 0.0
    last_added_time: float = 0.0

    # AR Reading Toggle (Thumbs-Up) & Air-Touch Fingertip Cursor state
    reading_enabled = True
    thumbs_up_hold_sec = 0.55
    thumbs_up_cooldown = 2.0
    thumbs_up_start: Optional[float] = None
    last_thumbs_toggle: float = 0.0

    ar_cursor: Optional[Tuple[int, int]] = None
    ar_hover_target: Optional[str] = None
    ar_hover_start: float = 0.0
    ar_hover_duration = 0.80   # Hold index fingertip over a pill/button for 0.8s to select
    ar_select_cooldown = 1.5
    last_ar_select_time: float = 0.0
    last_known_letter: Optional[str] = None

    # 4. Open Webcam Feed & Register Touch/Click Mouse Callback
    cap = initialize_webcam()
    window_name = "Dual-Dataset ASL Sign Language Translator (DUET AI-3202)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    ui_regions: Dict[str, Tuple[int, int, int, int]] = {}
    click_events: list = []

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_events.append((x, y))

    cv2.setMouseCallback(window_name, _on_mouse)

    print("\n[INFO] Interactive Controls:")
    print("   * Show THUMBS-UP (0.6s) or press 'r' to Toggle Letter Reading ON / OFF (AR Pointer Mode).")
    print("   * In AR Pointer Mode: Point Index Finger at any Sentence/Button for 0.8s to select & SPEAK.")
    print("   * Or Click/Touch any Suggested Sentence (or press '1', '2', '3') to select & SPEAK.")
    print("   * Press 'ENTER' / 'v' (or click SPEAK) to speak the constructed sentence aloud.")
    print("   * Press 'SPACE' to add space | 'BACKSPACE' to delete | 'a' to add letter | 'x' to clear.")
    print("   * Press 't' to Toggle Hold-to-Add Auto-Typing (ON / OFF).")
    print("   * Press 'g' (or click EVAL GRAPHS) to view all Model Evaluation Graphs for your demo.")
    print("   * Press 'd' to Toggle Dataset | 'm' to Switch Model (SVM <-> KNN) | 's' for Smoothing | 'q' to Quit.\n")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            fh, fw, _ = frame.shape

            # Active model and scaler based on current selection
            active_bundle = all_artifacts[current_dataset]
            active_model = active_bundle[current_model_type]
            active_scaler = active_bundle["scaler"]

            # Process hand landmarks
            landmarks = hand_tracker.process(frame)

            hand_detected = False
            raw_letter: Optional[str] = None
            confidence = 0.0
            now = time.time()
            thumbs_up_progress = 0.0

            if landmarks is not None:
                hand_detected = True
                hand_tracker.draw_landmarks(frame, landmarks)

                # Smooth AR Index Fingertip cursor (Landmark 8)
                tip = landmarks[8]
                tx = max(0, min(fw - 1, int(tip.x * fw)))
                ty = max(0, min(fh - 1, int(tip.y * fh)))
                if ar_cursor is None:
                    ar_cursor = (tx, ty)
                else:
                    ar_cursor = (
                        int(0.55 * ar_cursor[0] + 0.45 * tx),
                        int(0.55 * ar_cursor[1] + 0.45 * ty),
                    )

                # Thumbs-Up Gesture Toggle for Reading ON / OFF
                if (now - last_thumbs_toggle >= thumbs_up_cooldown) and is_thumbs_up(landmarks):
                    if thumbs_up_start is None:
                        thumbs_up_start = now
                    elapsed_tu = now - thumbs_up_start
                    thumbs_up_progress = min(1.0, elapsed_tu / thumbs_up_hold_sec)
                    if elapsed_tu >= thumbs_up_hold_sec:
                        reading_enabled = not reading_enabled
                        last_thumbs_toggle = now
                        thumbs_up_start = None
                        thumbs_up_progress = 0.0
                        smoother.clear()
                        ar_hover_target = None
                        mode_msg = "READING ON" if reading_enabled else "AR POINTER MODE (READING OFF)"
                        print(f"[*] Thumbs-Up detected -> Switched to {mode_msg}")
                else:
                    thumbs_up_start = None

                # Run ASL letter classification only when Reading is ON and not holding Thumbs-Up
                if reading_enabled and thumbs_up_progress == 0.0:
                    try:
                        scaled_features = normalize_live_landmarks(landmarks, scaler=active_scaler)
                        if hasattr(active_model, "predict_proba"):
                            proba_dist = active_model.predict_proba(scaled_features)[0]
                            pred_idx = int(np.argmax(proba_dist))
                            confidence = float(proba_dist[pred_idx])
                        else:
                            pred_idx = int(active_model.predict(scaled_features)[0])
                            confidence = 1.0
                        raw_letter = config.IDX_TO_LABEL.get(pred_idx, "?")
                    except Exception as e:
                        print(f"[!] Inference error: {e}")
            else:
                ar_cursor = None
                thumbs_up_start = None
                ar_hover_target = None

            # Update temporal smoothing queue when Reading is ON
            if reading_enabled and hand_detected and raw_letter:
                if temporal_smoothing_enabled:
                    active_letter, smooth_conf, is_stable = smoother.update(raw_letter, confidence)
                else:
                    active_letter = raw_letter
                    smooth_conf = confidence
                    is_stable = (confidence >= config.CONFIDENCE_THRESHOLD)
                if is_stable and active_letter and active_letter.isalpha():
                    last_known_letter = active_letter
            else:
                active_letter, smooth_conf, is_stable = smoother.update("?", 0.0)
                is_stable = False

            # Hold-to-Add Dwell Timer for Auto-Typing (only when Reading is ON)
            dwell_progress = 0.0
            if (
                reading_enabled
                and auto_type_enabled
                and hand_detected
                and is_stable
                and active_letter
                and active_letter.isalpha()
                and (now - last_added_time >= dwell_cooldown)
            ):
                if dwell_letter == active_letter:
                    elapsed = now - dwell_start_time
                    dwell_progress = min(1.0, elapsed / dwell_duration)
                    if elapsed >= dwell_duration:
                        sentence_engine.add_letter(active_letter)
                        last_added_time = now
                        dwell_start_time = now
                        dwell_progress = 0.0
                        print(f"[+] Added letter '{active_letter}' -> Sentence: '{sentence_engine.get_full_text()}'")
                else:
                    dwell_letter = active_letter
                    dwell_start_time = now
            else:
                if not reading_enabled or not hand_detected or not is_stable:
                    dwell_letter = None
                    dwell_start_time = now

            suggestion_seed = active_letter if (reading_enabled and hand_detected) else last_known_letter
            live_suggestions = sentence_engine.get_suggestions(suggestion_seed)

            # Shared UI action dispatcher (used by both Mouse Clicks and AR Fingertip Hover)
            open_graphs_requested = False

            def _execute_ui_action(region_id: str) -> None:
                nonlocal auto_type_enabled, open_graphs_requested
                if region_id.startswith("sug_"):
                    sug_idx = int(region_id.split("_")[1])
                    chosen = sentence_engine.select_suggestion(sug_idx, suggestion_seed)
                    if chosen:
                        print(f"[*] Selected suggestion: '{chosen}' (Speaking...)")
                        speech_engine.speak(chosen)
                elif region_id == "btn_add":
                    if hand_detected and active_letter and active_letter.isalpha():
                        sentence_engine.add_letter(active_letter)
                        print(f"[+] Added '{active_letter}' -> '{sentence_engine.get_full_text()}'")
                elif region_id == "btn_space":
                    sentence_engine.add_space()
                elif region_id == "btn_back":
                    sentence_engine.backspace()
                elif region_id == "btn_speak":
                    full_txt = sentence_engine.get_full_text()
                    if full_txt:
                        print(f"[*] Speaking sentence: '{full_txt}'")
                        speech_engine.speak(full_txt)
                elif region_id == "btn_clear":
                    sentence_engine.clear()
                    print("[*] Sentence cleared.")
                elif region_id == "btn_auto":
                    auto_type_enabled = not auto_type_enabled
                    print(f"[*] Auto-Type Dwell {'ENABLED' if auto_type_enabled else 'DISABLED'}.")
                elif region_id == "btn_graphs":
                    open_graphs_requested = True

            # Process mouse/touch clicks
            while click_events:
                cx, cy = click_events.pop(0)
                for region_id, (rx1, ry1, rx2, ry2) in ui_regions.items():
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        _execute_ui_action(region_id)
                        break

            # Process AR Fingertip Air-Hover selection when in AR Pointer Mode (Reading OFF)
            ar_hover_progress = 0.0
            if not reading_enabled and ar_cursor is not None and (now - last_ar_select_time >= ar_select_cooldown):
                ax, ay = ar_cursor
                hovered_id: Optional[str] = None
                for region_id, (rx1, ry1, rx2, ry2) in ui_regions.items():
                    if rx1 <= ax <= rx2 and ry1 <= ay <= ry2:
                        hovered_id = region_id
                        break
                if hovered_id is not None:
                    if ar_hover_target == hovered_id:
                        elapsed_ar = now - ar_hover_start
                        ar_hover_progress = min(1.0, elapsed_ar / ar_hover_duration)
                        if elapsed_ar >= ar_hover_duration:
                            _execute_ui_action(hovered_id)
                            last_ar_select_time = now
                            ar_hover_target = None
                            ar_hover_progress = 0.0
                    else:
                        ar_hover_target = hovered_id
                        ar_hover_start = now
                else:
                    ar_hover_target = None
            else:
                ar_hover_target = None

            fps = fps_counter.update()

            # Construct dynamic status badge: e.g. "[CUSTOM] SVM"
            dataset_tag = "KAGGLE" if current_dataset == "kaggle" else "CUSTOM"
            model_tag = "SVM" if current_model_type == "svm" else "KNN"
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
                hand_detected=hand_detected,
                sentence_text=sentence_engine.get_full_text(),
                suggestions=live_suggestions,
                dwell_progress=dwell_progress,
                is_speaking=speech_engine.is_speaking,
                auto_type_enabled=auto_type_enabled,
                ui_regions=ui_regions,
                reading_enabled=reading_enabled,
                thumbs_up_progress=thumbs_up_progress,
                ar_cursor=ar_cursor,
                ar_hover_target=ar_hover_target,
                ar_hover_progress=ar_hover_progress
            )

            cv2.imshow(window_name, frame)

            if open_graphs_requested:
                show_evaluation_graphs_viewer()

            # Keyboard event handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[*] Exit signal received. Shutting down...")
                break
            elif key == ord('r'):  # Manual fallback toggle for Reading ON / OFF
                reading_enabled = not reading_enabled
                smoother.clear()
                print(f"[*] Switched to {'READING ON' if reading_enabled else 'AR POINTER MODE'}.")
            elif key in (ord('1'), ord('2'), ord('3')):
                sug_idx = int(chr(key)) - 1
                chosen = sentence_engine.select_suggestion(sug_idx, suggestion_seed)
                if chosen:
                    print(f"[*] Selected suggestion [{sug_idx + 1}]: '{chosen}' (Speaking...)")
                    speech_engine.speak(chosen)
            elif key in (13, 10, ord('v')):  # ENTER or 'v' -> Speak sentence
                full_txt = sentence_engine.get_full_text()
                if full_txt:
                    print(f"[*] Speaking sentence: '{full_txt}'")
                    speech_engine.speak(full_txt)
            elif key == ord(' '):  # SPACE -> Add space between words
                sentence_engine.add_space()
            elif key == 8:  # BACKSPACE -> Delete last character
                sentence_engine.backspace()
            elif key == ord('a'):  # Manually add current letter
                if hand_detected and active_letter and active_letter.isalpha():
                    sentence_engine.add_letter(active_letter)
                    print(f"[+] Added '{active_letter}' -> '{sentence_engine.get_full_text()}'")
            elif key == ord('x'):  # Clear sentence
                sentence_engine.clear()
                print("[*] Sentence cleared.")
            elif key == ord('t'):  # Toggle auto-type dwell
                auto_type_enabled = not auto_type_enabled
                print(f"[*] Auto-Type Dwell {'ENABLED' if auto_type_enabled else 'DISABLED'}.")
            elif key == ord('g'):  # Open Evaluation & Graphs presentation viewer
                show_evaluation_graphs_viewer()
            elif key == ord('d'):  # Toggle Dataset Mode (Kaggle <-> Custom)
                if len(available_datasets) > 1:
                    current_dataset = "custom" if current_dataset == "kaggle" else "kaggle"
                    smoother.clear()
                    print(f"[*] Dataset Mode toggled to: {all_artifacts[current_dataset]['name'].upper()}")
            elif key == ord('m'):  # Toggle Classifier (SVM <-> KNN)
                current_model_type = "knn" if current_model_type == "svm" else "svm"
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
        speech_engine.close()
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.close()
        print("[+] Camera stream and HandTracker released cleanly.")
        print("[+] Application closed successfully.")


if __name__ == "__main__":
    run_realtime_translator()
