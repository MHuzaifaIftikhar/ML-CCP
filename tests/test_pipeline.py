"""
=============================================================================
AUTOMATED VERIFICATION SUITE
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
Tests end-to-end components: config, dataset, preprocessing, models,
temporal smoother, and AR HUD rendering.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from src.preprocessing import normalize_live_landmarks, load_and_validate_dataset
from src.utils import TemporalSmoother, draw_hud_overlay


def test_configuration():
    print("[TEST 1/7] Testing configuration...")
    assert len(config.LABELS) == 26, f"Expected 26 labels, got {len(config.LABELS)}"
    assert config.TOTAL_FEATURES == 63, f"Expected 63 features, got {config.TOTAL_FEATURES}"
    assert config.NUM_CLASSES == 26
    print("  --> PASS: Configuration verified.")


def test_dataset_integrity():
    print("[TEST 2/7] Testing dataset integrity...")
    assert config.RAW_CSV_PATH.exists(), f"Missing dataset at {config.RAW_CSV_PATH}"
    df = pd.read_csv(config.RAW_CSV_PATH)
    assert df.shape[1] == 64, f"Expected 64 columns (63 features + 1 label), got {df.shape[1]}"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values"
    unique_labels = sorted(df[config.TARGET_COLUMN].unique())
    assert unique_labels == config.LABELS, "Missing letters in dataset"
    print(f"  --> PASS: Dataset intact ({df.shape[0]} samples across all 26 letters).")


def test_artifacts_loading():
    print("[TEST 3/7] Testing saved model artifacts...")
    assert config.SVM_MODEL_PATH.exists(), "Missing svm_model.pkl"
    assert config.KNN_MODEL_PATH.exists(), "Missing knn_model.pkl"
    assert config.SCALER_PATH.exists(), "Missing scaler.pkl"

    svm = joblib.load(config.SVM_MODEL_PATH)
    knn = joblib.load(config.KNN_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    assert hasattr(svm, "predict_proba"), "SVM missing predict_proba"
    assert hasattr(knn, "predict"), "KNN missing predict"
    assert hasattr(scaler, "transform"), "Scaler missing transform"
    print("  --> PASS: All artifacts load cleanly.")
    return svm, knn, scaler


def test_landmark_normalization(scaler):
    print("[TEST 4/7] Testing live landmark normalization pipeline...")
    # Mock 21 3D points
    mock_landmarks = np.random.uniform(-0.5, 0.5, size=(21, 3)).astype(np.float32)
    mock_landmarks[0] = [0.0, 0.0, 0.0]  # Wrist origin

    scaled_feat = normalize_live_landmarks(mock_landmarks, scaler=scaler)
    assert scaled_feat.shape == (1, 63), f"Expected shape (1, 63), got {scaled_feat.shape}"
    assert not np.isnan(scaled_feat).any(), "NaN detected in scaled features"
    print("  --> PASS: Landmark normalization yields valid (1, 63) vector.")


def test_model_inference(svm, knn, scaler):
    print("[TEST 5/7] Testing live model inference...")
    mock_landmarks = np.zeros((21, 3), dtype=np.float32)
    # Simulate an 'L' shape: index pointing up, thumb sideways
    mock_landmarks[4] = [-0.5, -0.3, 0.0]   # Thumb tip
    mock_landmarks[8] = [-0.1, -0.9, 0.0]   # Index tip
    mock_landmarks[9] = [0.0, -0.4, 0.0]    # Middle MCP

    scaled_feat = normalize_live_landmarks(mock_landmarks, scaler=scaler)
    
    # SVM prediction
    svm_probs = svm.predict_proba(scaled_feat)[0]
    assert len(svm_probs) == 26
    assert np.isclose(np.sum(svm_probs), 1.0, atol=1e-3), "Probabilities do not sum to 1.0"
    pred_class_svm = config.IDX_TO_LABEL[int(np.argmax(svm_probs))]

    # KNN prediction
    pred_class_knn = config.IDX_TO_LABEL[int(knn.predict(scaled_feat)[0])]

    print(f"  --> PASS: SVM output letter '{pred_class_svm}' (conf: {np.max(svm_probs)*100:.1f}%), KNN output '{pred_class_knn}'.")


def test_temporal_smoother():
    print("[TEST 6/7] Testing Temporal Frame-Smoothing Engine...")
    smoother = TemporalSmoother(window_size=7, confidence_threshold=0.60, min_stable_count=4)

    # Push noisy frames initially
    letter, conf, is_stable = smoother.update('A', 0.50)
    assert not is_stable, "Should not be stable with low confidence"

    # Push 5 consistent high-confidence 'B' frames
    for _ in range(5):
        letter, conf, is_stable = smoother.update('B', 0.95)

    assert is_stable is True, "Should be marked stable after 5 consistent frames"
    assert letter == 'B', f"Expected majority vote 'B', got '{letter}'"
    assert len(smoother.get_recent_history()) <= 7
    print("  --> PASS: Temporal smoother filtered jitter and stabilized on majority vote.")


def test_hud_rendering():
    print("[TEST 7/7] Testing AR HUD overlay rendering...")
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = draw_hud_overlay(
        frame=dummy_frame,
        active_letter='A',
        confidence=0.98,
        is_stable=True,
        fps=29.8,
        recent_history=['A', 'A', 'A', 'A', 'A', 'A', 'A'],
        model_name="SVM (RBF Kernel)",
        hand_detected=True
    )
    assert rendered.shape == (720, 1280, 3), "Overlay altered frame shape"
    assert rendered.sum() > 0, "Overlay frame is unexpectedly empty"
    print("  --> PASS: AR HUD overlay rendered successfully.")


if __name__ == "__main__":
    print("=" * 70)
    print("STARTING ASL TRANSLATOR AUTOMATED VERIFICATION SUITE")
    print("=" * 70)
    test_configuration()
    test_dataset_integrity()
    svm, knn, scaler = test_artifacts_loading()
    test_landmark_normalization(scaler)
    test_model_inference(svm, knn, scaler)
    test_temporal_smoother()
    test_hud_rendering()
    print("=" * 70)
    print("ALL 7 VERIFICATION TESTS PASSED SUCCESSFULLY! [100% OK]")
    print("=" * 70)
