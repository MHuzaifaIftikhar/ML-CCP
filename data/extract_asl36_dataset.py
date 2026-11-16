"""
=============================================================================
ASL 36-CLASSES DATASET EXTRACTOR
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
Extracts 21 3D hand keypoints (63 features) from the ASL_Gestures_36_Classes
dataset (10 digits: 0-9 + 26 alphabet letters: A-Z = 36 classes total).

Features:
  - High-precision border padding (80px) for segmented hand images (100% detection)
  - Wrist-relative origin normalization (p_i - p_0) (Lab 01)
  - Euclidean palm scale invariance (/ ||p_9||) (Lab 01)
  - Exports to data/raw_csv/asl36_landmarks.csv in standard 64-column format
"""

import sys
import time
from pathlib import Path
from typing import List
import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.hand_tracker import HandTracker


def extract_asl36_landmarks(
    dataset_dir: Path = config.DATA_DIR / "ASL_Gestures_36_Classes",
    output_csv_path: Path = config.RAW_CSV_DIR / "asl36_landmarks.csv",
    pad_border: int = 80
) -> pd.DataFrame:
    """
    Processes all 36 gesture classes in the ASL_Gestures_36_Classes dataset.
    """
    dataset_dir = Path(dataset_dir)
    print("\n" + "=" * 76)
    print("EXTRACTING 3D HAND LANDMARKS: ASL 36-CLASSES GESTURE DATASET")
    print(f"Source: {dataset_dir}")
    print(f"Target: {output_csv_path}")
    print("=" * 76)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    tracker = HandTracker(min_detection_confidence=0.25)
    
    rows: List[List[float]] = []
    labels: List[str] = []
    class_stats = {}

    start_time = time.time()
    total_images_processed = 0

    # Process all class folders across 'train' and 'test'
    all_classes = [str(i) for i in range(10)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]

    for cls_name in all_classes:
        target_label = cls_name.upper()
        class_stats[target_label] = 0
        img_paths = []

        for split in ['train', 'test']:
            split_dir = dataset_dir / split / cls_name
            if split_dir.exists():
                img_paths.extend(list(split_dir.glob("*.jpeg")) + list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))

        print(f"[*] Processing Class '{target_label}' ({len(img_paths)} images)...", end="", flush=True)
        detected_in_class = 0

        for img_path in img_paths:
            total_images_processed += 1
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Apply border padding for segmented hand crops to give palm detector spatial context
            if pad_border > 0:
                img = cv2.copyMakeBorder(
                    img, pad_border, pad_border, pad_border, pad_border,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

            landmarks = tracker.process(img)
            if landmarks is not None and len(landmarks) == 21:
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                # Wrist origin normalization (Lab 01)
                wrist = coords[0].copy()
                translated = coords - wrist

                # Palm scale invariance (Lab 01)
                scale = np.linalg.norm(translated[9])
                if scale > 1e-6:
                    norm = translated / scale
                else:
                    norm = translated

                rows.append(norm.flatten().tolist())
                labels.append(target_label)
                detected_in_class += 1

        class_stats[target_label] = detected_in_class
        rate = (detected_in_class / max(1, len(img_paths))) * 100
        print(f" -> {detected_in_class}/{len(img_paths)} detected ({rate:.1f}%)")

    tracker.close()
    elapsed = time.time() - start_time

    if not rows:
        raise RuntimeError("No hand landmarks were detected in the dataset directory.")

    # Save to CSV
    df = pd.DataFrame(rows, columns=config.FEATURE_COLUMNS)
    df[config.TARGET_COLUMN] = labels

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print("\n" + "=" * 76)
    print(f"[+] ASL 36-CLASSES LANDMARK EXTRACTION COMPLETE!")
    print(f"    --> Total valid samples extracted: {len(df)}")
    print(f"    --> Classes detected: {df[config.TARGET_COLUMN].nunique()} / 36")
    print(f"    --> Total images examined: {total_images_processed}")
    print(f"    --> Processing time: {elapsed:.2f}s ({total_images_processed / max(0.01, elapsed):.1f} imgs/sec)")
    print(f"    --> Saved to: {output_csv_path}")
    print("=" * 76)

    return df


if __name__ == "__main__":
    extract_asl36_landmarks()
