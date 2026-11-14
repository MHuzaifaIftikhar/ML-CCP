"""
=============================================================================
KAGGLE DATASET IMPORTER & LANDMARK EXTRACTOR
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
This utility seamlessly converts external Kaggle datasets into our project's
standard 63-feature wrist-normalized landmark schema.

Supports two Kaggle formats:
  1. CSV Format: Pre-extracted landmarks from Kaggle (e.g. Rohit Paul, Kunal Gupta).
     Automatically maps columns and normalizes relative to the wrist.
  2. Image Directory Format: Raw ASL photos from Kaggle (e.g. Grassknoted ASL Alphabet).
     Runs our Universal HandTracker on the photos to extract authentic 3D keypoints!

Usage:
  python data/import_kaggle_csv.py --input "path/to/downloaded_kaggle_file.csv"
  python data/import_kaggle_csv.py --input "path/to/asl_alphabet_train_folder" --max-samples 150
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.hand_tracker import HandTracker


def import_kaggle_csv(input_csv_path: Path, output_csv_path: Path = config.KAGGLE_CSV_PATH) -> pd.DataFrame:
    """
    Imports and adapts a Kaggle landmark CSV file into our standard 64-column schema.
    """
    print(f"\n[*] Reading external Kaggle CSV from: {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    print(f"[+] Loaded {len(df)} rows with {len(df.columns)} columns.")

    # 1. Detect Label Column
    label_col = None
    for candidate in ['label', 'target', 'class', 'Letter', 'letter', 'Character']:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        # Check if first or last column is non-numeric
        if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
            label_col = df.columns[0]
        elif not np.issubdtype(df.iloc[:, -1].dtype, np.number):
            label_col = df.columns[-1]
        else:
            raise ValueError("Could not automatically detect the label/class column in the CSV.")

    print(f"[+] Detected target label column: '{label_col}'")
    labels = df[label_col].astype(str).str.strip().str.upper()

    # Filter only alphabetic letters A-Z
    valid_mask = labels.isin(config.LABELS)
    df = df[valid_mask].copy()
    labels = labels[valid_mask].values

    # 2. Extract 63 Coordinate Columns
    feature_df = df.drop(columns=[label_col])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 63:
        # Some 2D datasets only have 42 features (21 x 2). In that case, pad z with 0.0
        if len(numeric_cols) == 42:
            print("[!] Detected 42 features (2D landmarks). Up-converting to 63 features by setting z=0.0...")
            coords_2d = feature_df[numeric_cols].values.reshape(-1, 21, 2)
            coords_3d = np.zeros((len(df), 21, 3), dtype=np.float32)
            coords_3d[:, :, :2] = coords_2d
            flattened_features = coords_3d.reshape(len(df), 63)
        else:
            raise ValueError(f"Expected at least 42 or 63 numeric features, got {len(numeric_cols)}.")
    else:
        # Take the first 63 numeric landmark columns
        flattened_features = feature_df[numeric_cols[:63]].values.astype(np.float32)

    # 3. Apply Lab 01 Wrist Normalization & Scale Invariance
    print("[*] Normalizing coordinates relative to wrist origin (p_i - p_0)...")
    reshaped = flattened_features.reshape(-1, 21, 3)
    normalized_list = []

    for i in range(len(reshaped)):
        pose = reshaped[i]
        wrist = pose[0].copy()
        translated = pose - wrist
        palm_scale = np.linalg.norm(translated[9])
        if palm_scale > 1e-6:
            norm_pose = translated / palm_scale
        else:
            norm_pose = translated
        normalized_list.append(norm_pose.flatten())

    # 4. Save to project's kaggle_landmarks.csv
    out_df = pd.DataFrame(normalized_list, columns=config.FEATURE_COLUMNS)
    out_df[config.TARGET_COLUMN] = labels

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)

    print(f"\n[+] SUCCESS! Kaggle dataset successfully imported & standardized at:")
    print(f"    --> {output_csv_path}")
    print(f"    --> Total samples: {len(out_df)} across {out_df[config.TARGET_COLUMN].nunique()} letters.")
    return out_df


def import_kaggle_image_directory(
    image_dir: Path,
    output_csv_path: Path = config.KAGGLE_CSV_PATH,
    max_samples_per_class: int = 150
) -> pd.DataFrame:
    """
    Processes real Kaggle ASL photos (e.g. from the Grassknoted ASL Alphabet dataset),
    extracts 21 3D landmarks using MediaPipe, and exports to standard CSV.
    """
    image_dir = Path(image_dir)
    print(f"\n[*] Processing real Kaggle ASL photos from: {image_dir}...")
    tracker = HandTracker(min_detection_confidence=0.60)

    rows = []
    labels = []

    # Iterate over class folders A..Z
    for letter in config.LABELS:
        folder = image_dir / letter
        if not folder.exists():
            folder = image_dir / letter.lower()

        if not folder.exists():
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        print(f"  --> Processing letter '{letter}': {len(images)} images found...")

        count = 0
        for img_path in images:
            if count >= max_samples_per_class:
                break

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            landmarks = tracker.process(frame)
            if landmarks is not None and len(landmarks) == 21:
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                # Wrist origin normalization
                wrist = coords[0].copy()
                translated = coords - wrist
                scale = np.linalg.norm(translated[9])
                if scale > 1e-6:
                    norm = translated / scale
                else:
                    norm = translated

                rows.append(norm.flatten())
                labels.append(letter)
                count += 1

    tracker.close()

    if not rows:
        raise RuntimeError("No hand landmarks were detected in the provided image directory.")

    out_df = pd.DataFrame(rows, columns=config.FEATURE_COLUMNS)
    out_df[config.TARGET_COLUMN] = labels

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)

    print(f"\n[+] Extracted real MediaPipe landmarks from {len(out_df)} photos across {out_df[config.TARGET_COLUMN].nunique()} classes.")
    print(f"[+] Saved to: {output_csv_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Kaggle ASL dataset into project.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to downloaded Kaggle CSV file or folder of ASL images."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=150,
        help="Max samples per letter when extracting from an image folder (default: 150)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        import_kaggle_csv(input_path)
    elif input_path.is_dir():
        import_kaggle_image_directory(input_path, max_samples_per_class=args.max_samples)
    else:
        print(f"[!] Invalid path: {input_path}")
