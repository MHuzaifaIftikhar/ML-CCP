"""
=============================================================================
UTILITY MODULE: TEMPORAL SMOOTHING, AR HUD OVERLAY, AND EVALUATION VISUALIZATION
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
This module integrates:
  - Lab 13 / Lab 14: Real-time Temporal Frame-Smoothing Engine using collections.deque & Counter
  - Lab 13: Augmented Reality (AR) HUD visual overlays and MediaPipe landmark connections
  - Labs 06 & 08: Model evaluation metric plotting (Confusion Matrix heatmaps & comparison charts)
"""

import time
from collections import deque, Counter
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns

import config


# =============================================================================
# 1. TEMPORAL FRAME-SMOOTHING ENGINE (Lab 13 / Lab 14 CCA)
# =============================================================================
class TemporalSmoother:
    """
    Temporal Frame-Smoothing Engine using a sliding window FIFO queue (collections.deque)
    and majority voting (collections.Counter).
    
    Eliminates high-frequency prediction jitter/flickering caused by minor hand tremors,
    webcam sensor noise, or brief landmark occlusions.
    """
    def __init__(
        self,
        window_size: int = config.SMOOTHING_WINDOW_SIZE,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        min_stable_count: int = config.CONSECUTIVE_STABLE_FRAMES
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_stable_count = min_stable_count
        
        # FIFO queue storing recent raw predictions: tuples of (label_char, confidence)
        self.prediction_queue = deque(maxlen=self.window_size)
        
        self.current_stable_letter: Optional[str] = None
        self.current_stable_confidence: float = 0.0
        self.is_stable: bool = False

    def update(self, raw_letter: str, confidence: float) -> Tuple[Optional[str], float, bool]:
        """
        Pushes a new single-frame prediction into the queue and computes
        the smoothed majority vote.
        
        Args:
            raw_letter: Single-frame predicted letter (e.g., 'A').
            confidence: Associated prediction probability (0.0 to 1.0).
            
        Returns:
            Tuple[letter, confidence, is_stable]:
              - letter: Smoothed letter (or None if threshold not met).
              - confidence: Mean confidence of the dominant class in window.
              - is_stable: True if dominant class count >= min_stable_count.
        """
        # Filter out extremely noisy low-confidence frames
        if confidence >= (self.confidence_threshold * 0.75):
            self.prediction_queue.append((raw_letter, confidence))
        else:
            # Low confidence sample acts as soft decay
            self.prediction_queue.append(("?", confidence))

        if not self.prediction_queue:
            return None, 0.0, False

        # Extract only the letter candidates
        letters = [item[0] for item in self.prediction_queue if item[0] != "?"]
        
        if not letters:
            self.is_stable = False
            return None, 0.0, False

        # Majority voting via collections.Counter
        counts = Counter(letters)
        dominant_letter, dominant_count = counts.most_common(1)[0]

        # Calculate average confidence for the dominant letter
        confidences = [item[1] for item in self.prediction_queue if item[0] == dominant_letter]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Assess stability criteria
        if dominant_count >= self.min_stable_count and avg_confidence >= self.confidence_threshold:
            self.current_stable_letter = dominant_letter
            self.current_stable_confidence = avg_confidence
            self.is_stable = True
        else:
            self.is_stable = False

        return self.current_stable_letter, avg_confidence, self.is_stable

    def clear(self) -> None:
        """Clears the temporal history queue."""
        self.prediction_queue.clear()
        self.current_stable_letter = None
        self.current_stable_confidence = 0.0
        self.is_stable = False

    def get_recent_history(self) -> List[str]:
        """Returns list of letter symbols currently in sliding window."""
        return [item[0] for item in self.prediction_queue]


# =============================================================================
# 2. REAL-TIME FPS COUNTER
# =============================================================================
class FPSCounter:
    """Calculates instantaneous and exponential-moving-average FPS with warmup handling."""
    def __init__(self, smoothing: float = 0.85):
        self.prev_time = None
        self.fps = 30.0
        self.smoothing = smoothing

    def update(self) -> float:
        curr_time = time.time()
        if self.prev_time is None:
            self.prev_time = curr_time
            return self.fps
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        if delta > 0.001:
            instant_fps = 1.0 / delta
            # Clamp instant_fps to realistic webcam range (5 - 90 FPS) to filter out pauses
            instant_fps = max(5.0, min(90.0, instant_fps))
            self.fps = (self.smoothing * self.fps) + ((1.0 - self.smoothing) * instant_fps)
        return self.fps


# =============================================================================
# 2.5 GEOMETRIC AR GESTURE HELPERS (Thumbs-Up Mode Toggle)
# =============================================================================
def is_thumbs_up(landmarks: Any) -> bool:
    """
    Detects a clear 'Thumbs Up' gesture from 21 MediaPipe hand landmarks:
      - Thumb tip (4) is extended vertically well above thumb IP (3), MCP (2),
        and index knuckle (5/6).
      - All 4 fingers (index, middle, ring, pinky) are curled into the palm
        and well below the thumb tip.
    Scale-invariant relative to palm size (wrist 0 -> middle MCP 9).
    """
    if landmarks is None or len(landmarks) < 21:
        return False

    def _xy(idx: int) -> Tuple[float, float]:
        pt = landmarks[idx]
        if hasattr(pt, "x"):
            return float(pt.x), float(pt.y)
        return float(pt[0]), float(pt[1])

    wx, wy = _xy(0)
    m9x, m9y = _xy(9)
    palm_size = float(np.hypot(m9x - wx, m9y - wy))
    if palm_size < 1e-4:
        return False

    t4x, t4y = _xy(4)
    t3x, t3y = _xy(3)
    t2x, t2y = _xy(2)
    i5x, i5y = _xy(5)
    i6x, i6y = _xy(6)

    # 1. Thumb pointing strictly UP
    if not (t4y < t3y < t2y):
        return False
    if (t2y - t4y) < 0.42 * palm_size:
        return False
    if (min(i5y, i6y) - t4y) < 0.28 * palm_size:
        return False

    # 2. All 4 other fingers must be curled & well below thumb tip
    for tip_idx, mcp_idx in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tx, ty = _xy(tip_idx)
        mx, my = _xy(mcp_idx)
        # Fingertip must be noticeably below thumb tip
        if ty < t4y + 0.35 * palm_size:
            return False
        # Fingertip must stay compact near its knuckle (prevents 'L', 'D', 'G' false triggers)
        if np.hypot(tx - mx, ty - my) > 0.78 * palm_size:
            return False

    return True


# =============================================================================
# 3. AR VISUAL HUD OVERLAY RENDERER (Lab 13 & Lab 14 CCA)
# =============================================================================
def draw_hud_overlay(
    frame: np.ndarray,
    active_letter: Optional[str],
    confidence: float,
    is_stable: bool,
    fps: float,
    recent_history: List[str],
    model_name: str = "SVM (RBF Kernel)",
    hand_detected: bool = True,
    sentence_text: str = "",
    suggestions: Optional[List[str]] = None,
    dwell_progress: float = 0.0,
    is_speaking: bool = False,
    auto_type_enabled: bool = True,
    ui_regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    reading_enabled: bool = True,
    thumbs_up_progress: float = 0.0,
    ar_cursor: Optional[Tuple[int, int]] = None,
    ar_hover_target: Optional[str] = None,
    ar_hover_progress: float = 0.0
) -> np.ndarray:
    """
    Renders a high-contrast Augmented Reality (AR) HUD overlay on the webcam frame.

    Displays:
      - Top Banner: University, Course AI-3202, Project Name, [GRAPHS] button, Model, and FPS.
      - Top-Left Card: Active translated ASL letter, confidence bar, dwell progress, and queue.
      - Top-Right Panel: Touchable/clickable Sentence Suggestions ([1], [2], [3]).
      - Bottom Sentence Bar: Live constructed sentence + touch buttons (Add, Space, Back, Speak, Clear, Auto).
      - Bottom Footer: Keyboard hotkey instructions.
    """
    h, w, _ = frame.shape
    if ui_regions is not None:
        ui_regions.clear()

    # -------------------------------------------------------------------------
    # A. Top Header Banner (Semi-transparent dark bar)
    # -------------------------------------------------------------------------
    header_h = 65
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), config.COLOR_BACKGROUND, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.putText(
        frame,
        "DUET AI-3202 | COMPLEX COMPUTING ACTIVITY (CCA)",
        (20, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        config.COLOR_ACCENT_BLUE,
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "ASL Real-Time Sign Translator (Labs 01, 06, 08, 13)",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        config.COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    # Top-Right Clickable [EVAL GRAPHS] button
    if w >= 900:
        gx1, gy1, gx2, gy2 = w - 495, 14, w - 345, 50
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (45, 85, 140), -1)
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), config.COLOR_ACCENT_BLUE, 1)
        cv2.putText(frame, "[G] EVAL GRAPHS", (gx1 + 10, gy1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, config.COLOR_WHITE, 1, cv2.LINE_AA)
        if ui_regions is not None:
            ui_regions["btn_graphs"] = (gx1, gy1, gx2, gy2)

    # Model & FPS badges (Top Right)
    fps_text = f"FPS: {fps:.1f}"
    model_text = f"Model: {model_name}"
    cv2.putText(frame, model_text, (w - 325, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, config.COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (w - 140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, config.COLOR_PRIMARY_GREEN, 2, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # B. Main Prediction Card (Top Left floating glass panel)
    # -------------------------------------------------------------------------
    card_x1, card_y1 = 20, 80
    card_w, card_h = 280, 238
    card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h

    card_overlay = frame.copy()
    cv2.rectangle(card_overlay, (card_x1, card_y1), (card_x2, card_y2), config.COLOR_BACKGROUND, -1)
    cv2.addWeighted(card_overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), config.COLOR_PRIMARY_GREEN if reading_enabled else (0, 220, 255), 2)

    mode_badge = "READING: ON" if reading_enabled else "AR POINTER MODE"
    mode_badge_color = config.COLOR_PRIMARY_GREEN if reading_enabled else (0, 220, 255)
    cv2.putText(frame, "TRANSLATED SIGN", (card_x1 + 15, card_y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, config.COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, mode_badge, (card_x1 + 158, card_y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_badge_color, 1, cv2.LINE_AA)

    if not reading_enabled:
        display_letter = "AR"
        letter_color = (0, 220, 255)
    else:
        display_letter = active_letter if (active_letter and hand_detected) else "--"
        letter_color = config.COLOR_PRIMARY_GREEN if is_stable else config.COLOR_WARNING_YELLOW
        if not hand_detected:
            letter_color = config.COLOR_DANGER_RED

    text_size = cv2.getTextSize(display_letter, cv2.FONT_HERSHEY_DUPLEX, 3.0, 4)[0]
    letter_x = card_x1 + (card_w - text_size[0]) // 2
    letter_y = card_y1 + 118
    cv2.putText(frame, display_letter, (letter_x + 2, letter_y + 2), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, display_letter, (letter_x, letter_y), cv2.FONT_HERSHEY_DUPLEX, 3.0, letter_color, 4, cv2.LINE_AA)

    if thumbs_up_progress > 0:
        status_str = f"THUMBS UP: TOGGLING ({int(thumbs_up_progress*100)}%)"
        status_color = (0, 220, 255)
    elif not hand_detected:
        status_str = "NO HAND DETECTED"
        status_color = config.COLOR_DANGER_RED
    elif not reading_enabled:
        status_str = "POINT INDEX TO SELECT | THUMB=ON"
        status_color = (0, 220, 255)
    elif is_stable:
        status_str = f"CONFIDENT ({confidence*100:.1f}%)"
        status_color = config.COLOR_PRIMARY_GREEN
    else:
        status_str = f"STABILIZING... ({confidence*100:.1f}%)"
        status_color = config.COLOR_WARNING_YELLOW

    cv2.putText(frame, status_str, (card_x1 + 15, card_y1 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.43, status_color, 1, cv2.LINE_AA)

    # Confidence / Thumbs-Up progress bar
    bar_x1, bar_y1 = card_x1 + 15, card_y1 + 162
    bar_w, bar_h = card_w - 30, 12
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (60, 60, 60), -1)
    active_bar_val = thumbs_up_progress if thumbs_up_progress > 0 else (confidence if reading_enabled else 0.0)
    fill_w = int(bar_w * max(0.0, min(1.0, active_bar_val)))
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), status_color, -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), config.COLOR_WHITE, 1)

    # Hold-to-Add Dwell Progress Bar
    dwell_label = "Hold sign to auto-add:" if (auto_type_enabled and reading_enabled) else ("Thumbs-Up to resume reading" if not reading_enabled else "Auto-add OFF ([A] to add)")
    cv2.putText(frame, dwell_label, (card_x1 + 15, card_y1 + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    d_y1, d_h = card_y1 + 196, 8
    cv2.rectangle(frame, (bar_x1, d_y1), (bar_x1 + bar_w, d_y1 + d_h), (50, 50, 50), -1)
    if reading_enabled and auto_type_enabled and dwell_progress > 0:
        d_fill = int(bar_w * max(0.0, min(1.0, dwell_progress)))
        cv2.rectangle(frame, (bar_x1, d_y1), (bar_x1 + d_fill, d_y1 + d_h), (0, 220, 255), -1)

    # Sliding window queue breadcrumb display
    history_str = "Queue: [" + " ".join([item for item in recent_history[-7:]]) + "]"
    cv2.putText(frame, history_str, (card_x1 + 15, card_y1 + 225), cv2.FONT_HERSHEY_SIMPLEX, 0.43, config.COLOR_WHITE, 1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # C. Top-Right Touchable / AR-Selectable Sentence Suggestions Panel
    # -------------------------------------------------------------------------
    if suggestions is not None and w >= 850:
        sug_w = 415
        sug_x1 = w - sug_w - 20
        sug_y1 = 80
        sug_h = 225
        sug_x2 = w - 20

        sug_overlay = frame.copy()
        cv2.rectangle(sug_overlay, (sug_x1, sug_y1), (sug_x2, sug_y1 + sug_h), config.COLOR_BACKGROUND, -1)
        cv2.addWeighted(sug_overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (sug_x1, sug_y1), (sug_x2, sug_y1 + sug_h), config.COLOR_ACCENT_BLUE, 2)

        cv2.putText(
            frame,
            "SENTENCE SUGGESTIONS (AR Point / Touch / 1-3)",
            (sug_x1 + 14, sug_y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            config.COLOR_ACCENT_BLUE,
            1,
            cv2.LINE_AA
        )

        pill_y = sug_y1 + 40
        pill_h = 48
        pill_gap = 10
        for idx, phrase in enumerate(suggestions[:3]):
            py1 = pill_y + idx * (pill_h + pill_gap)
            py2 = py1 + pill_h
            px1, px2 = sug_x1 + 12, sug_x2 - 12

            region_key = f"sug_{idx}"
            is_hovered = (ar_hover_target == region_key)
            bg_col = (50, 95, 130) if is_hovered else (40, 52, 68)
            border_col = (0, 255, 255) if is_hovered else (config.COLOR_PRIMARY_GREEN if idx == 0 else (120, 145, 175))
            cv2.rectangle(frame, (px1, py1), (px2, py2), bg_col, -1)
            cv2.rectangle(frame, (px1, py1), (px2, py2), border_col, 2 if is_hovered else 1)

            display_phrase = phrase if len(phrase) <= 34 else phrase[:31] + "..."
            cv2.putText(
                frame,
                f"[{idx + 1}] {display_phrase}",
                (px1 + 12, py1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                config.COLOR_WHITE,
                1,
                cv2.LINE_AA
            )
            if ui_regions is not None:
                ui_regions[region_key] = (px1, py1, px2, py2)

    # -------------------------------------------------------------------------
    # D. Bottom Sentence Builder & Touch Controls Panel
    # -------------------------------------------------------------------------
    footer_h = 34
    if w >= 850:
        sent_h = 92
        sent_y2 = h - footer_h - 8
        sent_y1 = sent_y2 - sent_h
        sent_x1, sent_x2 = 20, w - 20

        sent_overlay = frame.copy()
        cv2.rectangle(sent_overlay, (sent_x1, sent_y1), (sent_x2, sent_y2), config.COLOR_BACKGROUND, -1)
        cv2.addWeighted(sent_overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (sent_x1, sent_y1), (sent_x2, sent_y2), config.COLOR_PRIMARY_GREEN if is_speaking else (90, 110, 130), 2)

        display_sent = sentence_text if sentence_text else "(Sign letters, Thumbs-Up for AR Pointer, or touch a suggestion...)"
        sent_color = config.COLOR_WHITE if sentence_text else (155, 165, 175)
        cv2.putText(
            frame,
            f"SENTENCE: {display_sent}",
            (sent_x1 + 15, sent_y1 + 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.66,
            sent_color,
            1,
            cv2.LINE_AA
        )

        if is_speaking:
            cv2.rectangle(frame, (sent_x2 - 155, sent_y1 + 8), (sent_x2 - 15, sent_y1 + 38), (0, 150, 75), -1)
            cv2.putText(frame, "SPEAKING...", (sent_x2 - 142, sent_y1 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_WHITE, 2, cv2.LINE_AA)

        buttons = [
            ("btn_add", "+ ADD [A]", (55, 75, 95), 115),
            ("btn_space", "SPACE", (55, 75, 95), 95),
            ("btn_back", "BACK", (85, 65, 55), 85),
            ("btn_speak", "SPEAK [ENTER]", (0, 145, 75), 155),
            ("btn_clear", "CLEAR [X]", (115, 45, 45), 115),
            ("btn_auto", f"AUTO: {'ON' if auto_type_enabled else 'OFF'} [T]", (60, 95, 120) if auto_type_enabled else (75, 75, 75), 165),
        ]
        bx = sent_x1 + 15
        by1, by2 = sent_y1 + 48, sent_y2 - 10
        for key_id, label, bg_color, bw in buttons:
            bx2 = bx + bw
            if bx2 < sent_x2 - 15:
                is_hovered = (ar_hover_target == key_id)
                cv2.rectangle(frame, (bx, by1), (bx2, by2), (70, 115, 150) if is_hovered else bg_color, -1)
                cv2.rectangle(frame, (bx, by1), (bx2, by2), (0, 255, 255) if is_hovered else (190, 200, 210), 2 if is_hovered else 1)
                cv2.putText(frame, label, (bx + 10, by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.47, config.COLOR_WHITE, 1, cv2.LINE_AA)
                if ui_regions is not None:
                    ui_regions[key_id] = (bx, by1, bx2, by2)
            bx = bx2 + 12

    # -------------------------------------------------------------------------
    # E. AR Fingertip Reticle & Dwell Selection Ring
    # -------------------------------------------------------------------------
    if ar_cursor is not None and not reading_enabled:
        cx, cy = ar_cursor
        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, (0, 230, 115), -1, cv2.LINE_AA)
        if ar_hover_target and ar_hover_progress > 0:
            end_angle = int(360 * max(0.0, min(1.0, ar_hover_progress)))
            cv2.ellipse(frame, (cx, cy), (22, 22), -90, 0, end_angle, (0, 255, 150), 3, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # F. Bottom Controls Banner
    # -------------------------------------------------------------------------
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (0, h - footer_h), (w, h), config.COLOR_BACKGROUND, -1)
    cv2.addWeighted(footer_overlay, 0.78, frame, 0.22, 0, frame)

    controls_text = "[Thumbs-Up / R]: Toggle Reading <-> AR Pointer | [1-3]: Pick | [ENTER]: Speak | [G]: Graphs | [M]: Model | [Q]: Quit"
    cv2.putText(
        frame,
        controls_text,
        (20, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        config.COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    return frame


def show_evaluation_graphs_viewer() -> None:
    """
    Opens an interactive presentation window allowing the user to flip through
    all generated model evaluation graphs and confusion matrices for demoing
    to a professor (Left/Right keys or on-screen Prev/Next buttons).
    """
    report_specs = [
        ("1/7: Model Benchmark Comparison (SVM vs KNN)", config.MODEL_COMPARISON_PLOT_PATH),
        ("2/7: Lab 06 SVM Confusion Matrix (Custom Dataset)", config.CONFUSION_MATRIX_SVM_PATH),
        ("3/7: Lab 08 KNN Confusion Matrix (Custom Dataset)", config.CONFUSION_MATRIX_KNN_PATH),
        ("4/7: Lab 06 SVM GridSearchCV Heatmap", config.REPORTS_DIR / "svm_gridsearch_heatmap.png"),
        ("5/7: Lab 08 KNN Hyperparameter Tuning Curves", config.REPORTS_DIR / "knn_tuning_curves.png"),
        ("6/7: Baseline vs Tuned Model Performance", config.REPORTS_DIR / "baseline_vs_tuned_comparison.png"),
        ("7/7: Cross-Dataset Benchmark (Custom vs Kaggle)", config.DATASET_COMPARISON_PLOT_PATH),
    ]
    available = [(title, p) for title, p in report_specs if p.exists()]
    if not available:
        print("[!] No evaluation graphs found in data/processed/reports/. Run src/train.py first.")
        return

    win_name = "Model Evaluation & Graphs Presentation (DUET AI-3202)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1180, 740)

    state = {"idx": 0, "prev_rect": (0, 0, 0, 0), "next_rect": (0, 0, 0, 0), "close_rect": (0, 0, 0, 0)}

    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        px1, py1, px2, py2 = state["prev_rect"]
        nx1, ny1, nx2, ny2 = state["next_rect"]
        cx1, cy1, cx2, cy2 = state["close_rect"]
        if px1 <= x <= px2 and py1 <= y <= py2:
            state["idx"] = (state["idx"] - 1) % len(available)
        elif nx1 <= x <= nx2 and ny1 <= y <= ny2:
            state["idx"] = (state["idx"] + 1) % len(available)
        elif cx1 <= x <= cx2 and cy1 <= y <= cy2:
            state["idx"] = -1

    cv2.setMouseCallback(win_name, _on_mouse)

    while state["idx"] >= 0:
        title, img_path = available[state["idx"]]
        chart = cv2.imread(str(img_path))
        canvas = np.full((740, 1180, 3), 24, dtype=np.uint8)

        # Header
        cv2.rectangle(canvas, (0, 0), (1180, 60), (20, 28, 38), -1)
        cv2.putText(canvas, title, (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.72, config.COLOR_ACCENT_BLUE, 1, cv2.LINE_AA)

        # Prev / Next / Close Buttons
        state["prev_rect"] = (760, 12, 880, 48)
        state["next_rect"] = (895, 12, 1015, 48)
        state["close_rect"] = (1030, 12, 1155, 48)

        cv2.rectangle(canvas, (760, 12), (880, 48), (55, 75, 95), -1)
        cv2.putText(canvas, "< PREV [A]", (772, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, config.COLOR_WHITE, 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (895, 12), (1015, 48), (55, 75, 95), -1)
        cv2.putText(canvas, "NEXT [D] >", (908, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, config.COLOR_WHITE, 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (1030, 12), (1155, 48), (130, 45, 45), -1)
        cv2.putText(canvas, "CLOSE [Q]", (1045, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, config.COLOR_WHITE, 1, cv2.LINE_AA)

        if chart is not None:
            ch, cw = chart.shape[:2]
            max_w, max_h = 1140, 650
            scale = min(max_w / cw, max_h / ch)
            nw, nh = int(cw * scale), int(ch * scale)
            resized = cv2.resize(chart, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = (1180 - nw) // 2
            oy = 70 + (650 - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q'), 27, ord('g'), ord('G')):
            break
        elif key in (ord('d'), ord('D'), ord('n'), ord('N'), 83, 2555904):
            state["idx"] = (state["idx"] + 1) % len(available)
        elif key in (ord('a'), ord('A'), ord('p'), ord('P'), 81, 2424832):
            state["idx"] = (state["idx"] - 1) % len(available)

    cv2.destroyWindow(win_name)


def draw_styled_hand_landmarks(
    frame: np.ndarray,
    landmarks_norm: Any,
    mp_drawing: Any,
    mp_hands: Any
) -> None:
    """
    Renders MediaPipe hand connections and keypoint dots with high-contrast custom colors.
    """
    if landmarks_norm is None or mp_drawing is None:
        return

    # Custom drawing specifications
    point_spec = mp_drawing.DrawingSpec(
        color=config.COLOR_LANDMARK_POINT,
        thickness=2,
        circle_radius=4
    )
    line_spec = mp_drawing.DrawingSpec(
        color=config.COLOR_LANDMARK_LINE,
        thickness=2,
        circle_radius=2
    )

    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks_norm,
        connections=mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=point_spec,
        connection_drawing_spec=line_spec
    )


# =============================================================================
# 4. MODEL EVALUATION VISUALIZATIONS (Labs 06 & 08 Alignment)
# =============================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = config.LABELS,
    title: str = "ASL Classification Confusion Matrix",
    save_path: Optional[Path] = None
) -> None:
    """
    Plots and saves an annotated confusion matrix heatmap using Seaborn and Matplotlib.
    
    Lab 06 & Lab 08 Alignment:
      - Visualizes classification performance across all 26 ASL classes.
    """
    plt.figure(figsize=(14, 11))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Sample Count'},
        linewidths=0.5,
        linecolor='lightgray'
    )
    plt.title(title, fontsize=16, pad=16, fontweight='bold')
    plt.xlabel("Predicted Sign Label", fontsize=12, labelpad=10)
    plt.ylabel("True Sign Label", fontsize=12, labelpad=10)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[+] Confusion matrix plot saved to: {save_path}")
    plt.close()


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = config.MODEL_COMPARISON_PLOT_PATH
) -> None:
    """
    Generates a publication-ready grouped bar chart comparing SVM (Lab 06)
    vs KNN (Lab 08) across key metrics: Accuracy, Precision, Recall, and F1-Score.
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = list(metrics_dict.keys())
    
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#2ca02c']  # Blue for SVM, Green for KNN

    for idx, model in enumerate(models):
        scores = [
            metrics_dict[model]["accuracy"],
            metrics_dict[model]["precision_macro"],
            metrics_dict[model]["recall_macro"],
            metrics_dict[model]["f1_macro"]
        ]
        offset = (idx - 0.5) * width
        rects = ax.bar(x + offset, scores, width, label=model, color=colors[idx % len(colors)], edgecolor='black')
        
        # Value labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height*100:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, fontweight='bold'
            )

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_title("Classifier Benchmark: SVM (Lab 06) vs KNN (Lab 08) on ASL Dataset", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight='semibold')
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[+] Model comparison chart saved to: {save_path}")
    plt.close()


def plot_knn_k_sensitivity(
    k_values: List[int],
    accuracies: List[float],
    save_path: Optional[Path] = None
) -> None:
    """
    Plots KNN classification accuracy as a function of k neighbors (Lab 08 Task 7).
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o', color='#d62728', linewidth=2, markersize=8)
    plt.title("Lab 08: KNN Accuracy vs Number of Neighbors (k)", fontsize=13, fontweight='bold')
    plt.xlabel("k (Number of Neighbors)", fontsize=11)
    plt.ylabel("Test Accuracy", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[+] KNN k-sensitivity plot saved to: {save_path}")
    plt.close()


def plot_cross_dataset_comparison(
    benchmark_data: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = config.DATASET_COMPARISON_PLOT_PATH
) -> None:
    """
    Plots a multi-bar comparison contrasting models trained on:
      1. Custom Lab Dataset (Webcam)
      2. Kaggle Benchmark Dataset (Clean)
    """
    labels = list(benchmark_data.keys())
    accuracies = [benchmark_data[k]["accuracy"] * 100 for k in labels]
    f1_scores = [benchmark_data[k]["f1_macro"] * 100 for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2b5c8f', edgecolor='black')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1 (%)', color='#36a282', edgecolor='black')

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Cross-Dataset Evaluation: Custom Lab Data vs Kaggle Benchmark Data', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='semibold')
    ax.set_ylim(80, 105)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[+] Cross-dataset comparison plot saved to: {save_path}")
    plt.close()
