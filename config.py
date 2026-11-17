"""
=============================================================================
CENTRAL CONFIGURATION MODULE
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
This module defines paths, hyperparameters, dataset specifications,
and real-time inference configurations across all lab integrations:
  - Lab 01: Data Preprocessing & Normalization
  - Lab 06: Support Vector Machine (SVM) Classifier
  - Lab 08: K-Nearest Neighbors (KNN) Classifier
  - Lab 13: ML x AR - Gesture Recognition & UI Overlay
"""

import os
from pathlib import Path
import string

# =============================================================================
# 1. DIRECTORY AND FILE PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV_DIR = DATA_DIR / "raw_csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = DATA_DIR / "processed" / "reports"

# Create directories if they do not already exist
RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectories for Dual-Mode Dataset Models (Custom Lab vs Kaggle Benchmark)
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"
KAGGLE_MODELS_DIR = MODELS_DIR / "kaggle"
CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset File Paths
RAW_CSV_PATH = RAW_CSV_DIR / "sign_landmarks.csv"
CUSTOM_CSV_PATH = RAW_CSV_DIR / "custom_landmarks.csv"
CLEANED_CUSTOM_CSV_PATH = RAW_CSV_DIR / "cleaned_custom_landmarks.csv"
KAGGLE_CSV_PATH = RAW_CSV_DIR / "kaggle_landmarks.csv"

# Preprocessed Data Splits
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_preprocessed.npz"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test_preprocessed.npz"

# Primary / Default Model Artifact Paths
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
KNN_MODEL_PATH = MODELS_DIR / "knn_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Custom Lab Model Artifact Paths
CUSTOM_SVM_PATH = CUSTOM_MODELS_DIR / "svm_model.pkl"
CUSTOM_KNN_PATH = CUSTOM_MODELS_DIR / "knn_model.pkl"
CUSTOM_SCALER_PATH = CUSTOM_MODELS_DIR / "scaler.pkl"

# Kaggle Benchmark Model Artifact Paths
KAGGLE_SVM_PATH = KAGGLE_MODELS_DIR / "svm_model.pkl"
KAGGLE_KNN_PATH = KAGGLE_MODELS_DIR / "knn_model.pkl"
KAGGLE_SCALER_PATH = KAGGLE_MODELS_DIR / "scaler.pkl"

# MediaPipe HandLandmarker Model
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_DOWNLOAD_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Evaluation Plot Paths
CONFUSION_MATRIX_SVM_PATH = REPORTS_DIR / "confusion_matrix_svm.png"
CONFUSION_MATRIX_KNN_PATH = REPORTS_DIR / "confusion_matrix_knn.png"
MODEL_COMPARISON_PLOT_PATH = REPORTS_DIR / "model_comparison.png"
DATASET_COMPARISON_PLOT_PATH = REPORTS_DIR / "dataset_mode_comparison.png"


# =============================================================================
# 2. ASL TARGET LABELS & FEATURE SPECIFICATIONS
# =============================================================================
# All 26 letters of the English alphabet for American Sign Language (A-Z)
LABELS = list(string.ascii_uppercase)  # ['A', 'B', 'C', ..., 'Z']
NUM_CLASSES = len(LABELS)               # 26 classes

# Bidirectional mapping between label characters and integer indices
LABEL_TO_IDX = {letter: idx for idx, letter in enumerate(LABELS)}
IDX_TO_LABEL = {idx: letter for idx, letter in enumerate(LABELS)}

# Hand Keypoint Feature Specifications:
# MediaPipe extracts 21 3D landmarks (x, y, z) per hand:
# 21 landmarks * 3 coordinates = 63 numerical features
NUM_LANDMARKS = 21
COORDINATES_PER_LANDMARK = 3
TOTAL_FEATURES = NUM_LANDMARKS * COORDINATES_PER_LANDMARK  # 63

# Generate column header names: x_0, y_0, z_0, ..., x_20, y_20, z_20
FEATURE_COLUMNS = []
for i in range(NUM_LANDMARKS):
    FEATURE_COLUMNS.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

TARGET_COLUMN = "label"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


# =============================================================================
# 3. MACHINE LEARNING HYPERPARAMETERS (Labs 01, 06, 08)
# =============================================================================
# Data Preprocessing & Splitting (Lab 01)
TEST_SIZE = 0.20           # 80% Training, 20% Testing
RANDOM_STATE = 42          # Seed for reproducibility
STRATIFY_DATA = True       # Maintain balanced class distribution across splits

# Support Vector Machine (Lab 06 - Optimal Tuned via GridSearchCV)
SVM_C = 20.0               # Tuned soft-margin regularization parameter
SVM_KERNEL = "rbf"         # Optimal Radial Basis Function kernel
SVM_GAMMA = 0.05           # Optimal kernel coefficient
SVM_PROBABILITY = True     # Platt scaling probability estimation for live AR HUD

# K-Nearest Neighbors (Lab 08 - Optimal Tuned via GridSearchCV)
KNN_NEIGHBORS = 1          # Optimal number of neighbors k (98.38% test accuracy)
KNN_METRIC = "manhattan"   # Optimal L1 Manhattan distance metric
KNN_WEIGHTS = "uniform"    # Optimal uniform neighbor weighting
KNN_K_RANGE = [1, 3, 5, 7, 9, 11]  # Values of k tested in tuning and sensitivity analysis


# =============================================================================
# 4. REAL-TIME INFERENCE & TEMPORAL SMOOTHING CONFIGURATION (Lab 13)
# =============================================================================
# Camera & Video Stream Settings
CAMERA_INDEX = 0           # Primary webcam index (0 is default, 1 for external)
FRAME_WIDTH = 1280         # High-definition 720p capture width in pixels (16:9 HD)
FRAME_HEIGHT = 720         # High-definition 720p capture height in pixels
TARGET_FPS = 30            # Target frame rate

# MediaPipe Hands Detector Parameters
MP_MAX_NUM_HANDS = 1       # Track single dominant hand for sign translation
MP_MIN_DETECTION_CONFIDENCE = 0.50  # Lowered from 0.70 for robust tracking in varied lighting
MP_MIN_TRACKING_CONFIDENCE = 0.50

# Temporal Smoothing Engine (Lab 13 / Lab 14 CCA)
SMOOTHING_WINDOW_SIZE = 7       # Sliding window size for majority voting queue
CONFIDENCE_THRESHOLD = 0.50    # Minimum probability required to accept prediction
CONSECUTIVE_STABLE_FRAMES = 3  # Responsive stabilization without sluggish delay

# UI Overlay Styling & Colors (BGR format for OpenCV)
COLOR_BACKGROUND = (20, 20, 20)        # Dark gray HUD background
COLOR_PRIMARY_GREEN = (0, 230, 115)     # Vibrant green for active recognized letter
COLOR_ACCENT_BLUE = (255, 170, 0)      # Cyan/Blue accent for HUD panels
COLOR_WARNING_YELLOW = (0, 215, 255)   # Amber yellow for low confidence/stabilizing
COLOR_DANGER_RED = (50, 50, 255)       # Red for alerts or missing hand
COLOR_WHITE = (245, 245, 245)          # Clean text white
COLOR_LANDMARK_POINT = (0, 255, 255)   # Yellow dots for joints
COLOR_LANDMARK_LINE = (0, 180, 255)    # Orange connection lines
