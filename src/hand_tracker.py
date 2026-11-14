"""
=============================================================================
UNIFIED HAND TRACKER MODULE (Cross-Version MediaPipe Support)
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
This module provides a robust, cross-version hand landmark extraction wrapper:
  - Supports modern MediaPipe Tasks API (mediapipe.tasks.python.vision.HandLandmarker)
    used in MediaPipe >= 0.10.14 and Python 3.14.
  - Supports legacy MediaPipe Solutions API (mediapipe.solutions.hands) used in older releases.
  - Automatically downloads and verifies hand_landmarker.task if missing.
  - Renders high-contrast skeletal connections and joint landmarks.
"""

import sys
import os
import urllib.request
from pathlib import Path
from typing import Optional, List, Tuple, Any
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

# Standard 21-keypoint hand skeletal connection pairs (MediaPipe topology)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]


class LandmarkPoint:
    """Lightweight 3D landmark representation with x, y, z properties."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"LandmarkPoint(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class HandTracker:
    """
    Universal Hand Tracker that seamlessly adapts to the available MediaPipe engine.
    """
    def __init__(
        self,
        max_num_hands: int = config.MP_MAX_NUM_HANDS,
        min_detection_confidence: float = config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MP_MIN_TRACKING_CONFIDENCE
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.backend = None
        self.detector = None

        self._initialize_detector()

    def _ensure_task_model_exists(self) -> Path:
        """Verifies or downloads hand_landmarker.task model artifact."""
        model_path = Path(config.HAND_LANDMARKER_MODEL_PATH)
        if not model_path.exists():
            print(f"[*] Downloading MediaPipe HandLandmarker model to: {model_path}...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(config.HAND_LANDMARKER_DOWNLOAD_URL, model_path)
                print(f"[+] Download complete ({model_path.stat().st_size / (1024*1024):.2f} MB).")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MediaPipe hand_landmarker.task from {config.HAND_LANDMARKER_DOWNLOAD_URL}: {e}"
                )
        return model_path

    def _initialize_detector(self):
        """Detects environment and sets up the appropriate MediaPipe backend."""
        import mediapipe as mp

        # Approach 1: Check for legacy solutions API (mediapipe < 0.10.14)
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self.detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            print("[+] Initialized MediaPipe Solutions backend.")
            return

        # Approach 2: Modern MediaPipe Tasks API (mediapipe >= 0.10.14 / Python 3.14)
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options

            task_model_path = self._ensure_task_model_exists()
            options = vision.HandLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(task_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.backend = "tasks"
            print(f"[+] Initialized MediaPipe Tasks HandLandmarker backend (model: {task_model_path.name}).")
            return
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MediaPipe HandLandmarker backend: {e}"
            )

    def process(self, frame_bgr: np.ndarray) -> Optional[List[LandmarkPoint]]:
        """
        Processes a BGR image frame and extracts 21 3D landmarks for the primary hand.
        
        Args:
            frame_bgr: Input frame in standard OpenCV BGR format.
            
        Returns:
            List[LandmarkPoint] of 21 keypoints, or None if no hand detected.
        """
        if self.backend == "solutions":
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                raw_landmarks = results.multi_hand_landmarks[0].landmark
                return [LandmarkPoint(lm.x, lm.y, lm.z) for lm in raw_landmarks]
            return None

        elif self.backend == "tasks":
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                raw_landmarks = results.hand_landmarks[0]
                return [LandmarkPoint(lm.x, lm.y, lm.z) for lm in raw_landmarks]
            return None

        return None

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[LandmarkPoint]],
        point_color: Tuple[int, int, int] = config.COLOR_LANDMARK_POINT,
        line_color: Tuple[int, int, int] = config.COLOR_LANDMARK_LINE
    ) -> None:
        """
        Draws 21 hand joints and connection lines directly onto the OpenCV frame.
        """
        if not landmarks or len(landmarks) != 21:
            return

        h, w, _ = frame.shape
        # Compute pixel coordinates
        pixel_points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pixel_points.append((px, py))

        # 1. Draw skeletal bones (connections)
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = pixel_points[start_idx]
            pt2 = pixel_points[end_idx]
            cv2.line(frame, pt1, pt2, line_color, 2, cv2.LINE_AA)

        # 2. Draw joint landmarks
        for idx, (px, py) in enumerate(pixel_points):
            # Wrist (0) and fingertips (4, 8, 12, 16, 20) are slightly larger
            radius = 5 if idx in [0, 4, 8, 12, 16, 20] else 4
            cv2.circle(frame, (px, py), radius, point_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

    def close(self):
        """Releases detector resources."""
        if self.detector is not None and hasattr(self.detector, "close"):
            self.detector.close()
