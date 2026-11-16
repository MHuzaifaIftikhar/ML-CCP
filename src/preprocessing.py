"""
=============================================================================
LAB 01 INTEGRATION: DATA PREPROCESSING & NORMALIZATION
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
This module implements the complete preprocessing pipeline aligned with Lab 01:
  - Task 1: Dataset Cleaning & Missing Value Detection/Imputation
  - Task 2: Target Label Encoding (A-Z to 0-25)
  - Task 3: Feature Scaling using StandardScaler (Z-score normalization: mean=0, std=1)
  - Task 4: Stratified 80/20 Train-Test Dataset Splitting
  - Real-Time Live Hand Landmark Normalization Engine for MediaPipe inference
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Ensure project root is in path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config


# =============================================================================
# 1. DATASET LOADING AND INTEGRITY VALIDATION (Lab 01 - Task 1)
# =============================================================================
def load_and_validate_dataset(csv_path: Union[str, Path] = config.RAW_CSV_PATH) -> pd.DataFrame:
    """
    Loads raw landmark dataset from CSV and validates structural integrity.
    
    Lab 01 Alignment:
      - Checks file existence and non-emptiness.
      - Validates that all 63 landmark coordinates (x_0..z_20) and the label column exist.
      - Detects and handles missing (NaN/null) values using statistical imputation or row removal.
    
    Args:
        csv_path: Path to the raw sign_landmarks.csv file.
        
    Returns:
        pd.DataFrame: Cleaned, validated dataframe.
    """
    csv_path = Path(csv_path)
    
    # Auto-generate dataset if not present to enable immediate, zero-friction execution
    if not csv_path.exists():
        print(f"[!] Warning: Dataset not found at {csv_path}. Initiating automatic generator...")
        try:
            from data.generate_dataset import create_asl_dataset
            create_asl_dataset(output_path=csv_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset not found at {csv_path} and automated generation failed: {e}"
            )
            
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded raw dataset with shape: {df.shape} from {csv_path}")
    
    # Check for required columns
    missing_cols = [col for col in config.ALL_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols[:5]}...")

    # Lab 01 - Task 1: Check for missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"[!] Detected {total_nulls} missing values in dataset. Applying imputation...")
        # Numeric features: impute with column median (less sensitive to outliers per Lab 01)
        for col in config.FEATURE_COLUMNS:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        # Drop rows where target label is missing
        df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
    else:
        print("[+] Data Cleaning: Zero missing values detected. Dataset is fully intact.")

    # Remove duplicates if any
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        print(f"[!] Removing {duplicates_count} duplicate rows.")
        df.drop_duplicates(inplace=True)
        
    return df


def filter_dataset_anomalies(
    df: pd.DataFrame,
    std_threshold: float = 2.2,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Lab 01 & Lab 14 Integration: Automated Anomaly Detection & Outlier Trimming.
    
    Filters out noisy, corrupted, or transitional hand gesture frames captured
    during live recording.
    
    For each ASL letter class:
      1. Computes the robust class centroid (median vector in 63D landmark space).
      2. Measures Euclidean distance of each sample to the centroid:
         d_i = || x_i - mu_c ||
      3. Identifies and trims outlier frames exceeding median(d) + (std_threshold * std(d)).
      
    Args:
        df: Input dataframe with 63 features and target label.
        std_threshold: Multiplier for standard deviation cutoff (default 2.2).
        verbose: If True, prints anomaly statistics.
        
    Returns:
        pd.DataFrame: Cleaned dataframe with transition anomalies removed.
    """
    cleaned_rows = []
    total_anomalies = 0

    for letter, group in df.groupby(config.TARGET_COLUMN):
        feature_matrix = group[config.FEATURE_COLUMNS].values
        # Robust centroid using coordinate medians
        centroid = np.median(feature_matrix, axis=0)
        # Euclidean distances to centroid
        distances = np.linalg.norm(feature_matrix - centroid, axis=1)
        
        # Adaptive threshold based on median and dispersion
        dist_median = np.median(distances)
        dist_std = np.std(distances)
        cutoff = dist_median + (std_threshold * dist_std)
        
        # Keep inliers
        inliers_mask = distances <= cutoff
        num_outliers = int(np.sum(~inliers_mask))
        total_anomalies += num_outliers
        
        cleaned_rows.append(group[inliers_mask])

    cleaned_df = pd.concat(cleaned_rows, ignore_index=True)
    # Shuffle to preserve uniform distribution
    cleaned_df = cleaned_df.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)
    
    if verbose:
        pct = (total_anomalies / max(1, len(df))) * 100
        print(f"[+] Anomaly Detection: Filtered out {total_anomalies} noisy/transition frames ({pct:.1f}%).")
        print(f"[+] Cleaned dataset retained {len(cleaned_df)} high-precision samples across all {cleaned_df[config.TARGET_COLUMN].nunique()} classes.")
        
    return cleaned_df


# =============================================================================
# 2. FEATURE EXTRACTION AND CATEGORICAL ENCODING (Lab 01 - Task 2)
# =============================================================================
def extract_features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Separates the numerical feature matrix X from the target label vector y,
    and applies Label Encoding to ensure standard integer class labels [0..25].
    
    Lab 01 Alignment:
      - Converts categorical letter labels ('A'..'Z') into integers (0..25).
      - Fits a LabelEncoder for consistent bidirectional encoding/decoding.
    
    Args:
        df: Input DataFrame containing 63 features and 1 label column.
        
    Returns:
        Tuple[X, y, label_encoder]:
          - X: Feature matrix of shape (N, 63).
          - y: Encoded target array of shape (N,).
          - label_encoder: Fitted sklearn LabelEncoder.
    """
    X = df[config.FEATURE_COLUMNS].values.astype(np.float32)
    raw_labels = df[config.TARGET_COLUMN].astype(str).str.strip().str.upper().values

    # Fit LabelEncoder on 36 classes (0-9, A-Z) or 26 alphabet classes (A-Z)
    label_encoder = LabelEncoder()
    if any(ch.isdigit() for ch in raw_labels) or len(np.unique(raw_labels)) > 26:
        label_encoder.fit(config.LABELS_36)
        class_desc = "36 classes (0-9 and A-Z mapped to 0-35)"
    else:
        label_encoder.fit(config.LABELS)
        class_desc = "26 classes (A-Z mapped to 0-25)"
    y = label_encoder.transform(raw_labels)

    print(f"[+] Features extracted: shape {X.shape}, dtype {X.dtype}")
    print(f"[+] Target encoded: {len(np.unique(y))} unique classes ({class_desc})")
    return X, y, label_encoder


# =============================================================================
# 3. STRATIFIED TRAIN-TEST SPLIT (Lab 01 - Task 4)
# =============================================================================
def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify: bool = config.STRATIFY_DATA
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions the dataset into 80% training and 20% testing subsets.
    
    Lab 01 Alignment:
      - Uses train_test_split to prevent overfitting and ensure reliable evaluation.
      - Employs stratified sampling to ensure equal class proportions in both sets.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )
    print(f"[+] Dataset Split: 80% Train ({X_train.shape[0]} samples), 20% Test ({X_test.shape[0]} samples)")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 4. FEATURE SCALING / STANDARDIZATION (Lab 01 - Task 3)
# =============================================================================
def fit_and_scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardizes features using Z-score normalization:
        z = (x - mu) / sigma
    ensuring each of the 63 hand landmark features has zero mean and unit variance.
    
    Lab 01 & ML Best Practices Alignment:
      - STRICT FEATURIZATION ORDERING: Scaler is fitted ONLY on X_train to prevent
        data leakage from the test distribution.
      - X_train is transformed using fit_transform.
      - X_test is transformed using the previously fitted scaler.
    
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[+] StandardScaler fitted on X_train (Mean={np.mean(X_train_scaled):.4f}, Std={np.std(X_train_scaled):.4f})")
    print(f"[+] X_test transformed using fitted parameters (Mean={np.mean(X_test_scaled):.4f}, Std={np.std(X_test_scaled):.4f})")
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# 5. PERSISTENCE OF PREPROCESSED ARTIFACTS
# =============================================================================
def save_preprocessing_artifacts(
    scaler: StandardScaler,
    scaler_path: Union[str, Path] = config.SCALER_PATH,
    X_train: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> None:
    """
    Persists the trained StandardScaler and preprocessed datasets to disk.
    """
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[+] Saved trained StandardScaler to: {scaler_path}")

    # Optionally persist numpy preprocessed arrays for fast reload
    if X_train is not None and y_train is not None:
        np.savez_compressed(
            config.PROCESSED_TRAIN_PATH,
            X=X_train,
            y=y_train
        )
        print(f"[+] Saved preprocessed training split to: {config.PROCESSED_TRAIN_PATH}")

    if X_test is not None and y_test is not None:
        np.savez_compressed(
            config.PROCESSED_TEST_PATH,
            X=X_test,
            y=y_test
        )
        print(f"[+] Saved preprocessed testing split to: {config.PROCESSED_TEST_PATH}")


def load_preprocessed_data(
    dataset_name: str = "custom",
    apply_anomaly_filter: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Loads and executes the full preprocessing pipeline for 'custom', 'kaggle', or 'asl36' dataset.
    
    Args:
        dataset_name: 'custom' (user webcam data), 'kaggle' (clean benchmark), or 'asl36' (36 classes).
        apply_anomaly_filter: If True and dataset_name == 'custom', applies automated outlier trimming.
        
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    if dataset_name.lower() in ["asl36", "36"]:
        csv_path = config.ASL36_CSV_PATH
        scaler_save_path = config.ASL36_SCALER_PATH
        print(f"\n[+] Preprocessing Mode: ASL 36-CLASSES GESTURE DATASET ({csv_path.name})")
    elif dataset_name.lower() == "kaggle":
        csv_path = config.KAGGLE_CSV_PATH
        scaler_save_path = config.KAGGLE_SCALER_PATH
        print(f"\n[+] Preprocessing Mode: KAGGLE BENCHMARK DATASET ({csv_path.name})")
    else:
        # Custom dataset
        if config.CLEANED_CUSTOM_CSV_PATH.exists() and not apply_anomaly_filter:
            csv_path = config.CLEANED_CUSTOM_CSV_PATH
        elif config.CUSTOM_CSV_PATH.exists():
            csv_path = config.CUSTOM_CSV_PATH
        else:
            csv_path = config.RAW_CSV_PATH
        scaler_save_path = config.CUSTOM_SCALER_PATH
        print(f"\n[+] Preprocessing Mode: CUSTOM LAB DATASET ({csv_path.name})")

    df = load_and_validate_dataset(csv_path)

    # Apply Anomaly Filtering on custom dataset if requested
    if dataset_name.lower() == "custom" and apply_anomaly_filter:
        print(f"[*] Applying automated anomaly detection & outlier trimming on custom dataset...")
        df = filter_dataset_anomalies(df, std_threshold=2.2, verbose=True)
        # Save cleaned version
        config.CLEANED_CUSTOM_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.CLEANED_CUSTOM_CSV_PATH, index=False)

    X, y, label_encoder = extract_features_and_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = fit_and_scale_features(X_train, X_test)

    # Save to dataset-specific model folder and default location
    save_preprocessing_artifacts(scaler, scaler_save_path, X_train_scaled, X_test_scaled, y_train, y_test)
    save_preprocessing_artifacts(scaler, config.SCALER_PATH)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# =============================================================================
# 6. REAL-TIME LANDMARK NORMALIZATION (Lab 01 & Lab 13 Integration)
# =============================================================================
def normalize_live_landmarks(
    raw_landmarks: Union[list, np.ndarray],
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Normalizes live hand keypoints extracted by MediaPipe Hands in real time:
      1. Translates all 21 keypoints relative to the wrist origin (Landmark 0):
         delta_p_i = p_i - p_0
      2. Normalizes scale by dividing by the palm scale distance (Landmark 0 to Landmark 9),
         rendering the feature vector invariant to user distance from camera.
      3. Flattens the 21 3D points into a 1D vector of 63 features.
      4. Standardizes using the trained StandardScaler (Lab 01 Z-score scaling).
      
    Args:
        raw_landmarks: Array-like of shape (21, 3) or list of landmark objects with x, y, z attributes.
        scaler: Fitted StandardScaler instance. If None, loaded from models/scaler.pkl.
        
    Returns:
        np.ndarray: Scaled feature vector of shape (1, 63) ready for model prediction.
    """
    # Convert MediaPipe landmark objects or list to numpy array
    if hasattr(raw_landmarks[0], 'x'):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks], dtype=np.float32)
    else:
        coords = np.array(raw_landmarks, dtype=np.float32)

    if coords.shape != (21, 3):
        raise ValueError(f"Expected landmark coordinates of shape (21, 3), got {coords.shape}")

    # Step 1: Translate relative to wrist (Landmark 0)
    wrist = coords[0].copy()
    translated = coords - wrist  # Landmark 0 becomes [0.0, 0.0, 0.0]

    # Step 2: Scale invariance (distance from wrist to middle finger knuckle MCP index 9)
    palm_scale = np.linalg.norm(translated[9])
    if palm_scale > 1e-6:
        # Scale canonical pose
        normalized_coords = translated / palm_scale
    else:
        normalized_coords = translated

    # Step 3: Flatten to 63-element feature vector
    features_1d = normalized_coords.flatten().reshape(1, -1)

    # Step 4: Apply trained StandardScaler (Lab 01 Z-score normalization)
    if scaler is None:
        if Path(config.SCALER_PATH).exists():
            scaler = joblib.load(config.SCALER_PATH)
        else:
            raise FileNotFoundError(
                f"StandardScaler artifact not found at {config.SCALER_PATH}. "
                "Please run 'python src/train.py' first to train and save the scaler."
            )

    scaled_features = scaler.transform(features_1d)
    return scaled_features


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING LAB 01 PREPROCESSING PIPELINE STANDALONE VERIFICATION")
    print("=" * 70)
    X_train, X_test, y_train, y_test, scaler = load_preprocessed_data()
    print(f"\n[OK] Preprocessing completed successfully!")
    print(f"     Train features: {X_train.shape}, Test features: {X_test.shape}")
    print(f"     Scaler saved to: {config.SCALER_PATH}")
