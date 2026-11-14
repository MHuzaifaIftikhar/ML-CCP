"""
=============================================================================
ASL HAND LANDMARK DATASET GENERATOR
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
This module generates a high-quality, physiologically accurate ASL landmark
dataset (sign_landmarks.csv) covering all 26 English letters (A-Z).
Each sample contains 21 3D hand keypoints (63 features) normalized relative
to the wrist origin (Landmark 0) with realistic sensor noise and joint variations.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path to load config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config


def get_canonical_pose(letter: str) -> np.ndarray:
    """
    Returns canonical 21 x 3 hand keypoints for a given ASL letter.
    Coordinates are defined relative to the wrist origin (0, 0, 0),
    where:
      - x: horizontal axis (left - to right +)
      - y: vertical axis (upwards is negative y in image coordinates)
      - z: depth axis (towards camera is negative z)
    """
    # 0: Wrist
    wrist = np.array([0.0, 0.0, 0.0])

    # Base MCP (knuckle) positions relative to wrist
    mcp_thumb = np.array([-0.20, -0.22, -0.02])
    mcp_index = np.array([-0.12, -0.45, 0.0])
    mcp_middle = np.array([0.0, -0.48, 0.0])
    mcp_ring = np.array([0.11, -0.45, 0.0])
    mcp_pinky = np.array([0.20, -0.40, 0.0])

    # Standard finger extension vectors [PIP, DIP, TIP]
    # Fully Extended Finger Templates
    ext_index = np.array([
        [-0.13, -0.65, -0.01],
        [-0.14, -0.80, -0.01],
        [-0.15, -0.95, -0.01]
    ])
    ext_middle = np.array([
        [0.0, -0.70, -0.01],
        [0.0, -0.88, -0.01],
        [0.0, -1.02, -0.01]
    ])
    ext_ring = np.array([
        [0.11, -0.65, -0.01],
        [0.12, -0.80, -0.01],
        [0.13, -0.95, -0.01]
    ])
    ext_pinky = np.array([
        [0.20, -0.55, -0.01],
        [0.21, -0.68, -0.01],
        [0.22, -0.80, -0.01]
    ])

    # Fully Curled (Fist) Finger Templates
    curled_index = np.array([
        [-0.12, -0.58, -0.07],
        [-0.10, -0.46, -0.12],
        [-0.08, -0.38, -0.08]
    ])
    curled_middle = np.array([
        [0.0, -0.60, -0.07],
        [0.0, -0.48, -0.12],
        [0.0, -0.40, -0.08]
    ])
    curled_ring = np.array([
        [0.10, -0.58, -0.07],
        [0.10, -0.46, -0.12],
        [0.08, -0.38, -0.08]
    ])
    curled_pinky = np.array([
        [0.18, -0.50, -0.07],
        [0.16, -0.40, -0.11],
        [0.14, -0.34, -0.07]
    ])

    # Thumb variants: [CMC, MCP, IP, TIP]
    # Thumb upright beside index (ASL 'A')
    thumb_upright = np.array([
        [-0.12, -0.15, -0.01],
        [-0.18, -0.28, -0.03],
        [-0.18, -0.44, -0.05],
        [-0.17, -0.58, -0.06]
    ])
    # Thumb tucked across palm (ASL 'B')
    thumb_tucked = np.array([
        [-0.10, -0.15, -0.02],
        [-0.05, -0.25, -0.05],
        [0.02, -0.32, -0.08],
        [0.06, -0.35, -0.08]
    ])
    # Thumb extended sideways in 'L' shape
    thumb_l = np.array([
        [-0.15, -0.18, -0.01],
        [-0.28, -0.28, -0.02],
        [-0.42, -0.35, -0.03],
        [-0.55, -0.38, -0.04]
    ])
    # Thumb folded across fist (ASL 'S')
    thumb_across_fist = np.array([
        [-0.12, -0.15, -0.02],
        [-0.08, -0.26, -0.06],
        [0.0, -0.35, -0.14],
        [0.08, -0.38, -0.13]
    ])
    # Thumb between index and middle (ASL 'T')
    thumb_t = np.array([
        [-0.12, -0.15, -0.02],
        [-0.10, -0.28, -0.05],
        [-0.06, -0.45, -0.12],
        [-0.04, -0.52, -0.08]
    ])

    # Construct full 21 landmarks based on ASL alphabet geometry
    letter = letter.upper()
    landmarks = np.zeros((21, 3))
    landmarks[0] = wrist
    landmarks[5] = mcp_index
    landmarks[9] = mcp_middle
    landmarks[13] = mcp_ring
    landmarks[17] = mcp_pinky

    if letter == 'A':
        # Fist with thumb resting alongside index
        landmarks[1:5] = thumb_upright
        landmarks[6:9] = curled_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'B':
        # 4 fingers straight up, thumb tucked across palm
        landmarks[1:5] = thumb_tucked
        landmarks[6:9] = ext_index
        landmarks[10:13] = ext_middle
        landmarks[14:17] = ext_ring
        landmarks[18:21] = ext_pinky

    elif letter == 'C':
        # Hand curved in C-shape
        c_curve = np.array([
            [-0.10, -0.55, -0.12],
            [-0.05, -0.62, -0.22],
            [0.02, -0.60, -0.28]
        ])
        landmarks[1:5] = np.array([
            [-0.15, -0.15, -0.02],
            [-0.22, -0.25, -0.08],
            [-0.20, -0.38, -0.18],
            [-0.12, -0.45, -0.25]
        ])
        landmarks[6:9] = c_curve + np.array([-0.08, 0.05, 0])
        landmarks[10:13] = c_curve
        landmarks[14:17] = c_curve + np.array([0.08, 0.05, 0])
        landmarks[18:21] = c_curve + np.array([0.16, 0.10, 0])

    elif letter == 'D':
        # Index straight up, other fingers curled touching thumb tip
        landmarks[1:5] = np.array([
            [-0.12, -0.15, -0.02],
            [-0.10, -0.26, -0.05],
            [-0.04, -0.38, -0.12],
            [0.01, -0.44, -0.11]
        ])
        landmarks[6:9] = ext_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'E':
        # All fingers tightly curled with fingertips resting on thumb
        landmarks[1:5] = np.array([
            [-0.12, -0.15, -0.02],
            [-0.06, -0.24, -0.06],
            [0.02, -0.30, -0.10],
            [0.08, -0.30, -0.10]
        ])
        e_curled = np.array([
            [-0.10, -0.52, -0.05],
            [-0.08, -0.42, -0.12],
            [-0.05, -0.34, -0.11]
        ])
        landmarks[6:9] = e_curled
        landmarks[10:13] = e_curled + np.array([0.06, 0.0, 0])
        landmarks[14:17] = e_curled + np.array([0.12, 0.0, 0])
        landmarks[18:21] = e_curled + np.array([0.18, 0.03, 0])

    elif letter == 'F':
        # Index and thumb tips touching (circle), middle/ring/pinky straight up
        landmarks[1:5] = np.array([
            [-0.12, -0.15, -0.02],
            [-0.16, -0.28, -0.05],
            [-0.14, -0.40, -0.10],
            [-0.08, -0.48, -0.10]
        ])
        landmarks[6:9] = np.array([
            [-0.12, -0.56, -0.06],
            [-0.10, -0.52, -0.10],
            [-0.08, -0.48, -0.10]
        ])
        landmarks[10:13] = ext_middle
        landmarks[14:17] = ext_ring
        landmarks[18:21] = ext_pinky

    elif letter == 'G':
        # Index pointing horizontally sideways, thumb parallel
        landmarks[1:5] = np.array([
            [-0.15, -0.18, -0.01],
            [-0.22, -0.28, -0.03],
            [-0.32, -0.38, -0.05],
            [-0.42, -0.42, -0.06]
        ])
        landmarks[6:9] = np.array([
            [-0.25, -0.50, -0.02],
            [-0.40, -0.52, -0.02],
            [-0.55, -0.52, -0.02]
        ])
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'H':
        # Index and middle fingers extended together horizontally
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = np.array([
            [-0.25, -0.50, -0.02],
            [-0.40, -0.52, -0.02],
            [-0.55, -0.52, -0.02]
        ])
        landmarks[10:13] = np.array([
            [-0.22, -0.54, -0.02],
            [-0.38, -0.56, -0.02],
            [-0.53, -0.56, -0.02]
        ])
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'I':
        # Pinky extended straight up, other fingers in fist, thumb folded over
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = curled_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = ext_pinky

    elif letter == 'J':
        # Pinky extended with slight J-curvature
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = curled_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = np.array([
            [0.22, -0.55, -0.01],
            [0.28, -0.66, 0.05],
            [0.35, -0.72, 0.12]
        ])

    elif letter == 'K':
        # Index straight up, middle tilted slightly forward, thumb between them
        landmarks[1:5] = np.array([
            [-0.12, -0.15, -0.02],
            [-0.10, -0.28, -0.04],
            [-0.05, -0.45, -0.06],
            [-0.02, -0.58, -0.08]
        ])
        landmarks[6:9] = ext_index
        landmarks[10:13] = np.array([
            [0.02, -0.65, -0.10],
            [0.04, -0.80, -0.18],
            [0.06, -0.92, -0.24]
        ])
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'L':
        # Index straight up, thumb extended perpendicular ('L' shape)
        landmarks[1:5] = thumb_l
        landmarks[6:9] = ext_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'M':
        # Thumb tucked under 3 fingers (index, middle, ring draped over thumb)
        landmarks[1:5] = np.array([
            [-0.10, -0.15, -0.02],
            [0.0, -0.25, -0.05],
            [0.10, -0.32, -0.08],
            [0.14, -0.35, -0.08]
        ])
        landmarks[6:9] = curled_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'N':
        # Thumb tucked under 2 fingers (index, middle draped over thumb)
        landmarks[1:5] = np.array([
            [-0.10, -0.15, -0.02],
            [-0.02, -0.25, -0.05],
            [0.05, -0.32, -0.08],
            [0.08, -0.35, -0.08]
        ])
        landmarks[6:9] = curled_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'O':
        # All fingertips touching thumb tip to form an 'O'
        landmarks[1:5] = np.array([
            [-0.12, -0.15, -0.02],
            [-0.14, -0.28, -0.08],
            [-0.08, -0.42, -0.18],
            [0.0, -0.48, -0.20]
        ])
        o_fingers = np.array([
            [-0.05, -0.58, -0.10],
            [-0.02, -0.55, -0.18],
            [0.0, -0.48, -0.20]
        ])
        landmarks[6:9] = o_fingers + np.array([-0.04, 0.0, 0])
        landmarks[10:13] = o_fingers
        landmarks[14:17] = o_fingers + np.array([0.04, 0.0, 0])
        landmarks[18:21] = o_fingers + np.array([0.08, 0.02, 0])

    elif letter == 'P':
        # Downward 'K' shape
        landmarks[1:5] = np.array([
            [-0.12, -0.15, -0.02],
            [-0.18, -0.25, -0.04],
            [-0.25, -0.32, -0.06],
            [-0.30, -0.36, -0.08]
        ])
        landmarks[6:9] = np.array([
            [-0.20, -0.45, -0.02],
            [-0.35, -0.42, -0.02],
            [-0.50, -0.38, -0.02]
        ])
        landmarks[10:13] = np.array([
            [-0.15, -0.42, -0.10],
            [-0.25, -0.35, -0.18],
            [-0.35, -0.28, -0.24]
        ])
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'Q':
        # Downward 'G' shape
        landmarks[1:5] = np.array([
            [-0.12, -0.15, -0.02],
            [-0.18, -0.25, -0.04],
            [-0.22, -0.32, -0.06],
            [-0.24, -0.38, -0.08]
        ])
        landmarks[6:9] = np.array([
            [-0.12, -0.48, -0.04],
            [-0.15, -0.40, -0.08],
            [-0.18, -0.32, -0.10]
        ])
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'R':
        # Index and middle fingers crossed, other fingers curled
        landmarks[1:5] = thumb_across_fist
        # Index crossed behind/over middle
        landmarks[6:9] = np.array([
            [-0.08, -0.65, -0.01],
            [-0.04, -0.80, 0.02],
            [0.02, -0.94, 0.04]
        ])
        # Middle crossed in front of index
        landmarks[10:13] = np.array([
            [-0.04, -0.68, -0.02],
            [-0.08, -0.83, -0.03],
            [-0.12, -0.96, -0.04]
        ])
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'S':
        # Tight fist with thumb crossed in front of fingers
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = curled_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'T':
        # Thumb between index and middle knuckles
        landmarks[1:5] = thumb_t
        landmarks[6:9] = curled_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'U':
        # Index and middle fingers extended straight up and pressed together
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = np.array([
            [-0.06, -0.65, -0.01],
            [-0.06, -0.80, -0.01],
            [-0.06, -0.95, -0.01]
        ])
        landmarks[10:13] = np.array([
            [0.02, -0.67, -0.01],
            [0.02, -0.82, -0.01],
            [0.02, -0.97, -0.01]
        ])
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'V':
        # Index and middle fingers extended apart in a 'V' (peace sign)
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = np.array([
            [-0.16, -0.65, -0.01],
            [-0.24, -0.80, -0.01],
            [-0.32, -0.95, -0.01]
        ])
        landmarks[10:13] = np.array([
            [0.08, -0.68, -0.01],
            [0.16, -0.84, -0.01],
            [0.24, -0.98, -0.01]
        ])
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'W':
        # Index, middle, ring extended spread out in a 'W'
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = np.array([
            [-0.18, -0.65, -0.01],
            [-0.26, -0.80, -0.01],
            [-0.32, -0.95, -0.01]
        ])
        landmarks[10:13] = ext_middle
        landmarks[14:17] = np.array([
            [0.18, -0.65, -0.01],
            [0.26, -0.80, -0.01],
            [0.32, -0.95, -0.01]
        ])
        landmarks[18:21] = curled_pinky

    elif letter == 'X':
        # Index finger bent into a hook/claw shape
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = np.array([
            [-0.12, -0.62, -0.02],
            [-0.10, -0.68, -0.12],
            [-0.08, -0.58, -0.18]
        ])
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    elif letter == 'Y':
        # Thumb and pinky extended wide, middle three curled ('hang loose' / shaka)
        landmarks[1:5] = np.array([
            [-0.15, -0.18, -0.01],
            [-0.30, -0.28, -0.02],
            [-0.45, -0.36, -0.03],
            [-0.60, -0.42, -0.04]
        ])
        landmarks[6:9] = curled_index
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = np.array([
            [0.26, -0.54, -0.01],
            [0.38, -0.65, -0.01],
            [0.50, -0.75, -0.01]
        ])

    elif letter == 'Z':
        # Index extended pointing forward in z-tracing pose
        landmarks[1:5] = thumb_across_fist
        landmarks[6:9] = np.array([
            [-0.12, -0.58, -0.08],
            [-0.13, -0.72, -0.20],
            [-0.14, -0.86, -0.32]
        ])
        landmarks[10:13] = curled_middle
        landmarks[14:17] = curled_ring
        landmarks[18:21] = curled_pinky

    else:
        raise ValueError(f"Unsupported ASL letter '{letter}'. Must be between A and Z.")

    return landmarks


def generate_augmented_samples(
    canonical_pose: np.ndarray,
    n_samples: int = 100,
    noise_std: float = 0.015,
    scale_range: tuple = (0.92, 1.08),
    angle_deg_range: tuple = (-8.0, 8.0),
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Synthesizes realistic variations of the canonical pose using:
      1. Global 3D rotation jitter (wrist orientation variance).
      2. Uniform hand scale variation.
      3. Landmark-specific Gaussian sensor noise.
    """
    if rng is None:
        rng = np.random.default_rng(config.RANDOM_STATE)

    samples = []
    for _ in range(n_samples):
        # 1. Hand scale variation
        scale = rng.uniform(scale_range[0], scale_range[1])
        pose = canonical_pose * scale

        # 2. Random 3D rotation jitter (Euler angles around x, y, z)
        theta_x = np.radians(rng.uniform(angle_deg_range[0], angle_deg_range[1]))
        theta_y = np.radians(rng.uniform(angle_deg_range[0], angle_deg_range[1]))
        theta_z = np.radians(rng.uniform(angle_deg_range[0], angle_deg_range[1]))

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
        pose = pose @ R.T

        # 3. Gaussian sensor noise on keypoints (wrist stays at 0, 0, 0)
        noise = rng.normal(0, noise_std, size=pose.shape)
        noise[0] = 0.0  # Wrist remains origin
        noisy_pose = pose + noise

        # Flatten 21 x 3 to 63 features
        samples.append(noisy_pose.flatten())

    return np.array(samples)


def create_asl_dataset(
    output_path: Path = config.RAW_CSV_PATH,
    samples_per_class: int = 120,
    random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """
    Generates the complete 26-letter ASL landmark dataset and saves to CSV.
    """
    print(f"[*] Generating synthetic ASL landmark dataset for all {config.NUM_CLASSES} letters (A-Z)...")
    rng = np.random.default_rng(random_state)

    all_features = []
    all_labels = []

    for letter in config.LABELS:
        canonical = get_canonical_pose(letter)
        class_samples = generate_augmented_samples(
            canonical_pose=canonical,
            n_samples=samples_per_class,
            rng=rng
        )
        all_features.append(class_samples)
        all_labels.extend([letter] * samples_per_class)

    X = np.vstack(all_features)
    y = np.array(all_labels)

    # Build DataFrame
    df = pd.DataFrame(X, columns=config.FEATURE_COLUMNS)
    df[config.TARGET_COLUMN] = y

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[+] Dataset successfully created at: {output_path}")
    print(f"[+] Total samples: {len(df)} ({samples_per_class} per class across 26 letters)")
    print(f"[+] Feature count: {len(config.FEATURE_COLUMNS)} | Label column: '{config.TARGET_COLUMN}'")
    return df


if __name__ == "__main__":
    create_asl_dataset()
