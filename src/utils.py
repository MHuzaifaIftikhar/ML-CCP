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
    hand_detected: bool = True
) -> np.ndarray:
    """
    Renders a high-contrast Augmented Reality (AR) HUD overlay on the webcam frame.
    
    Displays:
      - Top Banner: University, Course AI-3202, Project Name, Active Model, and FPS.
      - Left Panel: Prominent active translated ASL letter in bold green with card background.
      - Confidence Meter: Dynamic horizontal progress bar.
      - Status Indicator: 'TRANSLATING (STABLE)', 'STABILIZING...', or 'NO HAND DETECTED'.
      - Queue Breadcrumbs: Recent 7-frame sliding window history.
      - Bottom Banner: Keyboard hotkey instructions.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # -------------------------------------------------------------------------
    # A. Top Header Banner (Semi-transparent dark bar)
    # -------------------------------------------------------------------------
    header_h = 65
    cv2.rectangle(overlay, (0, 0), (w, header_h), config.COLOR_BACKGROUND, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & Course Alignment
    cv2.putText(
        frame,
        "DUET AI-3202 | COMPLEX COMPUTING ACTIVITY (CCA)",
        (20, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        config.COLOR_ACCENT_BLUE,
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "ASL Real-Time Sign Translator (Labs 01, 06, 08, 13)",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    # Model & FPS badges (Top Right)
    fps_text = f"FPS: {fps:.1f}"
    model_text = f"Model: {model_name}"
    cv2.putText(
        frame,
        model_text,
        (w - 320, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.COLOR_WHITE,
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        fps_text,
        (w - 140, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        config.COLOR_PRIMARY_GREEN,
        2,
        cv2.LINE_AA
    )

    # -------------------------------------------------------------------------
    # B. Main Prediction Card (Top Left floating glass panel)
    # -------------------------------------------------------------------------
    card_x1, card_y1 = 20, 80
    card_w, card_h = 280, 220
    card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h

    # Card background
    card_overlay = frame.copy()
    cv2.rectangle(card_overlay, (card_x1, card_y1), (card_x2, card_y2), config.COLOR_BACKGROUND, -1)
    cv2.addWeighted(card_overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), config.COLOR_ACCENT_BLUE, 2)

    cv2.putText(
        frame,
        "TRANSLATED SIGN",
        (card_x1 + 15, card_y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    # Active Letter Display
    display_letter = active_letter if (active_letter and hand_detected) else "--"
    letter_color = config.COLOR_PRIMARY_GREEN if is_stable else config.COLOR_WARNING_YELLOW
    if not hand_detected:
        letter_color = config.COLOR_DANGER_RED

    # Large bold letter font with shadow for contrast
    text_size = cv2.getTextSize(display_letter, cv2.FONT_HERSHEY_DUPLEX, 3.2, 4)[0]
    letter_x = card_x1 + (card_w - text_size[0]) // 2
    letter_y = card_y1 + 125
    # Shadow
    cv2.putText(frame, display_letter, (letter_x + 2, letter_y + 2), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0, 0, 0), 6, cv2.LINE_AA)
    # Primary letter
    cv2.putText(frame, display_letter, (letter_x, letter_y), cv2.FONT_HERSHEY_DUPLEX, 3.2, letter_color, 4, cv2.LINE_AA)

    # Status text under letter
    if not hand_detected:
        status_str = "NO HAND DETECTED"
        status_color = config.COLOR_DANGER_RED
    elif is_stable:
        status_str = f"CONFIDENT ({confidence*100:.1f}%)"
        status_color = config.COLOR_PRIMARY_GREEN
    else:
        status_str = f"STABILIZING... ({confidence*100:.1f}%)"
        status_color = config.COLOR_WARNING_YELLOW

    cv2.putText(
        frame,
        status_str,
        (card_x1 + 15, card_y1 + 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        status_color,
        1,
        cv2.LINE_AA
    )

    # Confidence progress bar
    bar_x1, bar_y1 = card_x1 + 15, card_y1 + 175
    bar_w, bar_h = card_w - 30, 14
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * max(0.0, min(1.0, confidence)))
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), letter_color, -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), config.COLOR_WHITE, 1)

    # Sliding window queue breadcrumb display
    history_str = "Queue: [" + " ".join([h for h in recent_history[-7:]]) + "]"
    cv2.putText(
        frame,
        history_str,
        (card_x1 + 15, card_y1 + 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        config.COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    # -------------------------------------------------------------------------
    # C. Bottom Controls Banner
    # -------------------------------------------------------------------------
    footer_h = 35
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (0, h - footer_h), (w, h), config.COLOR_BACKGROUND, -1)
    cv2.addWeighted(footer_overlay, 0.75, frame, 0.25, 0, frame)

    controls_text = "[Q]: Quit  |  [D]: Toggle Dataset (Kaggle <-> Custom)  |  [M]: Switch Model  |  [S]: Smoothing  |  [C]: Clear"
    cv2.putText(
        frame,
        controls_text,
        (25, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        config.COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    return frame


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
