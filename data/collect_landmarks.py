"""
=============================================================================
INTERACTIVE WEBCAM ASL DATASET COLLECTOR
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering and Technology, Batch 2023
=============================================================================
This tool allows students and researchers to collect real-world custom ASL
hand landmark datasets directly via webcam using MediaPipe Hands (Lab 13 Task 2).

Features:
  - High-Definition 1280x720 16:9 AR HUD layout (dynamically responsive)
  - Interactive class selection (cycle A-Z or jump to any letter)
  - Automated 100-sample batch recording with 3-second pose countdown ([ENTER])
  - Continuous live recording with 0.5s spacebar debounce ([SPACE])
  - Automatic wrist-origin normalization and scale invariance (Lab 01)
  - Auto-repeat safe double-confirmation for letter clearing ([C])
  - Live on-screen sample counter and dataset synchronization across CSVs
  - Real-time Pose Coach tips tailored for difficult letters (G, H, J, K, N, O, U, V)
"""

import sys
import time
import shutil
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils import FPSCounter
from src.hand_tracker import HandTracker


ASL_POSE_GUIDELINES = {
    'A': "Fist with thumb resting along outer side of index finger (thumb points UP).",
    'B': "4 fingers pointing straight UP together; thumb folded flat across palm.",
    'C': "Fingers curved forming a 'C' shape. Turn hand slightly sideways to camera.",
    'D': "Index pointing straight UP; thumb and other 3 fingers touch forming an 'O'.",
    'E': "All 4 fingertips curled tightly down touching the top of thumb (claw).",
    'F': "Index and thumb tips touching in circle (OK sign); other 3 fingers UP.",
    'G': "Index pointing SIDEWAYS (horizontal); thumb parallel above it. Turn hand sideways to camera!",
    'H': "Index AND Middle pointing SIDEWAYS together. Turn hand sideways! (2 fingers vs 1 in G)",
    'I': "Pinky pointing straight UP; other 3 fingers curled into a fist with thumb across.",
    'J': "Hold STATIC terminal pose of J: Pinky UP, wrist angled/turned ~30 deg inward.",
    'K': "Index pointing straight UP, Middle angled forward/up, thumb touching middle knuckle. Angle 15 deg.",
    'L': "Index pointing UP, thumb pointing SIDEWAYS, forming an 'L' shape.",
    'M': "Fist with thumb tucked under 3 fingers (index, middle, ring).",
    'N': "Fist with thumb tucked under 2 fingers (index, middle). Tilt slightly so thumb shows!",
    'O': "All 5 fingertips curved touching thumb tip to form an open circle 'O'. Angle so hole shows!",
    'P': "Like 'K' but pointing downwards diagonally.",
    'Q': "Like 'G' but pointing downwards diagonally.",
    'R': "Index and Middle fingers pointing UP and crossed over each other.",
    'S': "Tight fist with thumb folded ACROSS the front of all 4 fingers (over knuckles).",
    'T': "Fist with thumb poking between index and middle finger.",
    'U': "Index and Middle fingers pointing straight UP, pressed tightly TOGETHER!",
    'V': "Index and Middle fingers pointing straight UP, spread apart in an open PEACE SIGN!",
    'W': "Index, Middle, and Ring fingers pointing UP and spread apart in a 'W'.",
    'X': "Fist with index finger bent into a hook.",
    'Y': "Thumb and Pinky extended outwards (hang loose / phone sign).",
    'Z': "Static terminal pose of Z: Index finger pointed, hand angled ~15 deg."
}


class ASLDatasetCollector:
    """
    Interactive application for recording ASL hand landmark samples from webcam.
    """
    def __init__(
        self,
        output_csv_path: Path = config.RAW_CSV_PATH,
        samples_per_batch: int = 100
    ):
        self.output_csv_path = Path(output_csv_path)
        self.samples_per_batch = samples_per_batch
        self.current_letter_idx = 0  # Starts at 'A'
        
        # Load existing dataset if present to show current counts
        self.existing_counts = {letter: 0 for letter in config.LABELS}
        self._load_existing_sample_counts()

        # State machine: 'READY', 'COUNTDOWN', 'RECORDING'
        self.state = "READY"
        self.countdown_start_time = 0.0
        self.countdown_seconds = 3
        self.batch_target_mode = False
        self.target_batch_count = 100
        self.recorded_in_current_batch = 0
        self.total_recorded_this_session = 0
        self.batch_landmarks: List[List[float]] = []
        self.data_modified = False
        self.last_toggle_time = 0.0
        self.pending_clear = False
        self.pending_clear_time = 0.0

        # Universal HandTracker (supports Tasks and Solutions API)
        self.hand_tracker = HandTracker()
        self.fps_counter = FPSCounter()

    def _load_existing_sample_counts(self):
        """Loads sample distribution across all dataset files with multi-source fallback."""
        target_files = [
            self.output_csv_path,
            config.CUSTOM_CSV_PATH,
            config.RAW_CSV_PATH,
            config.CLEANED_CUSTOM_CSV_PATH,
            config.RAW_CSV_DIR / "custom_landmarks_backup.csv"
        ]

        for p in target_files:
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if config.TARGET_COLUMN in df.columns:
                        counts = df[config.TARGET_COLUMN].value_counts().to_dict()
                        for letter in config.LABELS:
                            self.existing_counts[letter] = max(
                                self.existing_counts[letter],
                                counts.get(letter, 0)
                            )
                except Exception as e:
                    print(f"[!] Warning reading existing CSV {p.name}: {e}")

        # Synchronize custom_landmarks.csv and sign_landmarks.csv if any file had missing classes
        min_count = min(self.existing_counts.values()) if self.existing_counts else 0
        backup_file = config.RAW_CSV_DIR / "custom_landmarks_backup.csv"
        if min_count == 0 and backup_file.exists():
            print("[*] Re-synchronizing missing classes from backup dataset...")
            try:
                backup_df = pd.read_csv(backup_file)
                for p in [self.output_csv_path, config.CUSTOM_CSV_PATH, config.RAW_CSV_PATH]:
                    backup_df.to_csv(p, index=False)
                counts = backup_df[config.TARGET_COLUMN].value_counts().to_dict()
                for letter in config.LABELS:
                    self.existing_counts[letter] = counts.get(letter, 0)
                print("[+] Successfully synchronized all 26 classes across dataset files.")
            except Exception as e:
                print(f"[!] Error syncing from backup: {e}")

    @property
    def current_letter(self) -> str:
        return config.LABELS[self.current_letter_idx]

    def clear_current_letter(self):
        """Wipes existing samples for the currently selected letter after safety backup."""
        target_files = set([self.output_csv_path, config.CUSTOM_CSV_PATH, config.RAW_CSV_PATH])
        old_count = self.existing_counts.get(self.current_letter, 0)

        # 1. Create a timestamped safety backup first so user never loses data permanently
        backup_dir = config.RAW_CSV_DIR
        timestamp = int(time.time())
        safety_backup = backup_dir / f"backup_before_clearing_{self.current_letter}_{timestamp}.csv"
        if config.CUSTOM_CSV_PATH.exists():
            shutil.copy(config.CUSTOM_CSV_PATH, safety_backup)
            print(f"[+] Created safety backup at: {safety_backup.name}")

        # 2. Filter out current letter
        for p in target_files:
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if config.TARGET_COLUMN in df.columns:
                        df_clean = df[df[config.TARGET_COLUMN] != self.current_letter]
                        df_clean.to_csv(p, index=False)
                except Exception as e:
                    print(f"[!] Error clearing letter from {p.name}: {e}")

        self.existing_counts[self.current_letter] = 0
        self.data_modified = True
        print(f"\n[!] CLEARED all {old_count} samples for sign '{self.current_letter}'.")
        print(f"[*] Ready to record fresh, clean poses for '{self.current_letter}'! Press [SPACE] or [ENTER].")

    def _normalize_raw_landmarks(self, landmark_list) -> List[float]:
        """
        Applies Lab 01 normalization:
          1. Wrist translation (Landmark 0 becomes 0, 0, 0)
          2. Scale invariance (divided by distance to Landmark 9)
          3. Flattens to 63 features
        """
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)
        wrist = coords[0].copy()
        translated = coords - wrist

        palm_scale = np.linalg.norm(translated[9])
        if palm_scale > 1e-6:
            normalized = translated / palm_scale
        else:
            normalized = translated

        return normalized.flatten().tolist()

    def save_batch(self):
        """Saves collected batch and synchronizes across raw CSV paths."""
        if not self.batch_landmarks:
            return

        new_df = pd.DataFrame(self.batch_landmarks, columns=config.FEATURE_COLUMNS)
        new_df[config.TARGET_COLUMN] = self.current_letter

        target_files = set([self.output_csv_path, config.CUSTOM_CSV_PATH, config.RAW_CSV_PATH])
        for p in target_files:
            if p.exists():
                new_df.to_csv(p, mode='a', header=False, index=False)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                new_df.to_csv(p, mode='w', header=True, index=False)

        count_added = len(self.batch_landmarks)
        self.existing_counts[self.current_letter] += count_added
        self.total_recorded_this_session += count_added
        self.data_modified = True
        print(f"[+] Saved {count_added} samples for sign '{self.current_letter}'. Total on disk: {self.existing_counts[self.current_letter]}")
        self.batch_landmarks.clear()

    def draw_collector_hud(self, frame: np.ndarray, hand_present: bool) -> np.ndarray:
        """Draws recording status, target letter, progress, live pose coach, and keyboard guide."""
        h, w, _ = frame.shape

        # 1. Top Header Banner
        header_h = 75
        cv2.rectangle(frame, (0, 0), (w, header_h), config.COLOR_BACKGROUND, -1)
        
        # Responsive title to prevent any text collisions
        if w >= 900:
            title_text = f"ASL DATASET RECORDER (Lab 13 Task 2) | Active Letter: '{self.current_letter}'"
            title_scale = 0.65
        else:
            title_text = f"ASL RECORDER | Sign: '{self.current_letter}'"
            title_scale = 0.55

        cv2.putText(
            frame,
            title_text,
            (25, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            title_scale,
            config.COLOR_ACCENT_BLUE,
            1,
            cv2.LINE_AA
        )

        # FPS Badge (Top Right)
        fps = self.fps_counter.update()
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (w - 140, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            config.COLOR_PRIMARY_GREEN,
            1,
            cv2.LINE_AA
        )

        current_count = self.existing_counts[self.current_letter]
        sub_info = f"Saved on Disk for '{self.current_letter}': {current_count} samples  |  Session: +{self.total_recorded_this_session}  |  [C]: Clear Letter"
        cv2.putText(
            frame,
            sub_info,
            (25, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50 if w >= 900 else 0.42,
            config.COLOR_WHITE,
            1,
            cv2.LINE_AA
        )

        # 2. State & Action HUD Card (Top Left)
        card_w = 400 if w >= 900 else 340
        card_h = 195
        card_x, card_y = 25, 90
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), config.COLOR_BACKGROUND, -1)
        
        border_color = config.COLOR_PRIMARY_GREEN if self.state == "RECORDING" else config.COLOR_ACCENT_BLUE
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), border_color, 2)

        if self.state == "READY":
            status_text = "|| PAUSED (READY)"
            status_color = config.COLOR_WARNING_YELLOW
            instr_text = "Press [SPACE] for continuous live record"
            sub_text = "Or press [ENTER] for 100-sample countdown batch"
            prog_label = "Ready to record"
            fill_pct = 0.0

        elif self.state == "COUNTDOWN":
            elapsed = time.time() - self.countdown_start_time
            remaining = max(1, int(self.countdown_seconds - elapsed) + 1)
            status_text = f"COUNTDOWN... ({remaining}s)"
            status_color = config.COLOR_WARNING_YELLOW
            instr_text = f"Hold pose for sign '{self.current_letter}'"
            sub_text = f"Recording starts automatically in {remaining}s!"
            prog_label = f"Starting in {remaining}s"
            fill_pct = min(1.0, elapsed / self.countdown_seconds)

        elif self.state == "RECORDING":
            if hand_present:
                if self.batch_target_mode:
                    status_text = f"● BATCH RECORDING ({self.recorded_in_current_batch}/100)"
                    instr_text = "Hold pose steady with minor micro-movements"
                    sub_text = "Batch saves automatically upon reaching 100!"
                    prog_label = f"Batch: {self.recorded_in_current_batch} / 100"
                    fill_pct = min(1.0, self.recorded_in_current_batch / 100.0)
                else:
                    status_text = f"● RECORDING LIVE (+{self.recorded_in_current_batch} samples)"
                    instr_text = "Press [SPACE] to PAUSE recording"
                    sub_text = "Subtle hand rotations enhance dataset variety"
                    prog_label = f"Samples: +{self.recorded_in_current_batch}"
                    fill_pct = min(1.0, (self.recorded_in_current_batch % 100) / 100.0)
                status_color = config.COLOR_PRIMARY_GREEN
            else:
                status_text = "● RECORDING (WAITING FOR HAND)"
                status_color = config.COLOR_WARNING_YELLOW
                instr_text = "Bring your hand into camera view!"
                sub_text = "Recording captures automatically once tracked"
                prog_label = "Waiting for hand detection..."
                fill_pct = 0.0

        cv2.putText(frame, status_text, (card_x + 18, card_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, instr_text, (card_x + 18, card_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, config.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, sub_text, (card_x + 18, card_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        # Progress bar
        bar_w = card_w - 36
        bar_x = card_x + 18
        bar_y = card_y + 98
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), (50, 50, 50), -1)
        fill_w = int(bar_w * fill_pct)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 16), config.COLOR_PRIMARY_GREEN, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), config.COLOR_WHITE, 1)

        cv2.putText(frame, prog_label, (bar_x, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # Hand detection status indicator
        hand_status = "[OK] Hand Tracking: ACTIVE" if hand_present else "[!] Hand Tracking: NO HAND IN VIEW"
        hand_color = config.COLOR_PRIMARY_GREEN if hand_present else config.COLOR_DANGER_RED
        cv2.putText(frame, hand_status, (card_x + 18, card_y + 175), cv2.FONT_HERSHEY_DUPLEX, 0.46, hand_color, 1, cv2.LINE_AA)

        # 3. Emergency Clear Confirmation Warning Alert (if pending)
        if self.pending_clear:
            alert_w = 480
            alert_h = 44
            cv2.rectangle(frame, (25, card_y + card_h + 15), (25 + alert_w, card_y + card_h + 15 + alert_h), (0, 0, 180), -1)
            cv2.rectangle(frame, (25, card_y + card_h + 15), (25 + alert_w, card_y + card_h + 15 + alert_h), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"CONFIRM WIPE: Press [C] again to DELETE '{self.current_letter}'!",
                (35, card_y + card_h + 43),
                cv2.FONT_HERSHEY_DUPLEX,
                0.46,
                config.COLOR_WHITE,
                1,
                cv2.LINE_AA
            )

        # 4. Live Pose Coach Tip Card (Bottom Banner)
        tip_text = ASL_POSE_GUIDELINES.get(self.current_letter, "")
        cv2.rectangle(frame, (0, h - 75), (w, h - 35), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"POSE COACH [{self.current_letter}]: {tip_text}",
            (25, h - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48 if w >= 900 else 0.40,
            (0, 235, 255),  # Yellow
            1,
            cv2.LINE_AA
        )

        # 5. Bottom Hotkey Guide
        cv2.rectangle(frame, (0, h - 35), (w, h), config.COLOR_BACKGROUND, -1)
        nav_text = "[SPACE] Record/Pause  |  [ENTER] 100-Batch  |  [C] Clear  |  [A-Z] Jump  |  [[]/[]] Prev/Next  |  [ESC] Exit"
        cv2.putText(
            frame,
            nav_text,
            (25, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 if w >= 900 else 0.38,
            config.COLOR_WHITE,
            1,
            cv2.LINE_AA
        )

        return frame

    def run(self):
        """Main webcam collection loop."""
        print("\n" + "=" * 76)
        print("LAUNCHING INTERACTIVE ASL DATASET RECORDER")
        print(f"Target file: {self.output_csv_path}")
        print("[!] Controls:")
        print("    - [SPACE]: Toggle continuous LIVE recording ON / PAUSE")
        print("    - [ENTER]: 100-sample automated batch (with 3-second countdown)")
        print("    - [C]: Wipe/delete existing samples for active letter (double-press)")
        print("    - [A-Z]: Jump directly to ANY letter (all 26 letters work!)")
        print("    - [ [ / ] ]: Previous / Next letter")
        print("    - [ESC]: Save and exit cleanly")
        print("=" * 76)

        # Open camera with DirectShow on Windows for instant capture
        if sys.platform == "win32":
            cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(config.CAMERA_INDEX)
        else:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("[!] Error: Could not open webcam for recording.")
            return

        # Request 1280x720 HD resolution matching config
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        window_name = "ASL Dataset Collector (DUET AI-3202)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

        try:
            while cap.isOpened():
                # Check if window was closed with mouse [X]
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                success, frame = cap.read()
                if not success or frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                landmarks = self.hand_tracker.process(frame)

                hand_present = False
                if landmarks is not None:
                    hand_present = True
                    self.hand_tracker.draw_landmarks(frame, landmarks)

                # State Transitions: Countdown -> Recording
                if self.state == "COUNTDOWN":
                    elapsed = time.time() - self.countdown_start_time
                    if elapsed >= self.countdown_seconds:
                        self.state = "RECORDING"
                        self.recorded_in_current_batch = 0
                        self.batch_landmarks.clear()

                # Recording State: Capture landmarks
                elif self.state == "RECORDING":
                    if hand_present and landmarks is not None:
                        features_63 = self._normalize_raw_landmarks(landmarks)
                        self.batch_landmarks.append(features_63)
                        self.recorded_in_current_batch += 1

                        # Incremental auto-save every 25 samples
                        if len(self.batch_landmarks) >= 25:
                            self.save_batch()

                        # Check batch target completion
                        if self.batch_target_mode and self.recorded_in_current_batch >= self.target_batch_count:
                            self.save_batch()
                            self.state = "READY"
                            self.batch_target_mode = False
                            print(f"[+] Batch complete! Saved 100 samples for '{self.current_letter}'. Total on disk: {self.existing_counts[self.current_letter]}")

                # Timeout pending clear confirmation after 4 seconds
                if self.pending_clear and (time.time() - self.pending_clear_time > 4.0):
                    self.pending_clear = False

                frame = self.draw_collector_hud(frame, hand_present)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to exit
                    if self.batch_landmarks:
                        self.save_batch()
                    break

                elif key == ord(' '):  # SPACE: Toggle Live Continuous Recording
                    now = time.time()
                    if now - self.last_toggle_time > 0.5:  # 500ms debounce
                        self.last_toggle_time = now
                        self.pending_clear = False
                        if self.state == "RECORDING":
                            self.save_batch()
                            self.state = "READY"
                            self.batch_target_mode = False
                            print(f"[*] Paused recording for sign '{self.current_letter}'. Press [SPACE] to resume.")
                        else:
                            self.state = "RECORDING"
                            self.batch_target_mode = False
                            self.recorded_in_current_batch = 0
                            print(f"[*] Started continuous recording for sign '{self.current_letter}'! Press [SPACE] to pause.")

                elif key in [13, 10]:  # ENTER: 100-sample automated batch mode
                    now = time.time()
                    if now - self.last_toggle_time > 0.5:
                        self.last_toggle_time = now
                        self.pending_clear = False
                        if self.batch_landmarks:
                            self.save_batch()
                        self.state = "COUNTDOWN"
                        self.countdown_start_time = time.time()
                        self.batch_target_mode = True
                        self.recorded_in_current_batch = 0
                        print(f"[*] Starting 100-sample batch for sign '{self.current_letter}'! Hold pose ready...")

                elif key in [ord('c'), ord('C'), 8, 127]:  # 'C', Backspace, or Delete
                    now = time.time()
                    if self.pending_clear and (0.4 <= now - self.pending_clear_time <= 4.0):
                        if self.batch_landmarks:
                            self.batch_landmarks.clear()
                        self.clear_current_letter()
                        self.state = "READY"
                        self.pending_clear = False
                    elif not self.pending_clear or (now - self.pending_clear_time > 4.0):
                        self.pending_clear = True
                        self.pending_clear_time = now
                        print(f"\n[!] WARNING: Press [C] again within 4 seconds to CONFIRM wiping '{self.current_letter}'!")

                elif key in [ord('['), ord(','), ord('<')]:  # Previous letter
                    self.pending_clear = False
                    if self.batch_landmarks:
                        self.save_batch()
                    self.current_letter_idx = (self.current_letter_idx - 1) % len(config.LABELS)
                    self.state = "READY"
                    self.batch_target_mode = False
                    print(f"[*] Switched to target letter '{self.current_letter}'.")

                elif key in [ord(']'), ord('.'), ord('>')]:  # Next letter
                    self.pending_clear = False
                    if self.batch_landmarks:
                        self.save_batch()
                    self.current_letter_idx = (self.current_letter_idx + 1) % len(config.LABELS)
                    self.state = "READY"
                    self.batch_target_mode = False
                    print(f"[*] Switched to target letter '{self.current_letter}'.")

                else:
                    # Check if user pressed ANY letter key (A-Z)
                    self.pending_clear = False
                    try:
                        char = chr(key).upper()
                        if char in config.LABELS:
                            if self.batch_landmarks:
                                self.save_batch()
                            self.current_letter_idx = config.LABEL_TO_IDX[char]
                            self.state = "READY"
                            self.batch_target_mode = False
                            print(f"[*] Jumped to target letter '{self.current_letter}'.")
                    except ValueError:
                        pass

        finally:
            # Always save any remaining samples
            if self.batch_landmarks:
                self.save_batch()

            cap.release()
            cv2.destroyAllWindows()
            self.hand_tracker.close()
            print("\n[+] Dataset collector session finished.")

            # If dataset was modified, automatically re-run anomaly cleaning & sync
            if self.data_modified:
                print("[*] Synchronizing and re-cleaning custom dataset...")
                try:
                    from src.preprocessing import filter_dataset_anomalies
                    raw_df = pd.read_csv(config.CUSTOM_CSV_PATH)
                    clean_df = filter_dataset_anomalies(raw_df, std_threshold=2.2, verbose=True)
                    clean_df.to_csv(config.CLEANED_CUSTOM_CSV_PATH, index=False)
                    print(f"[+] Successfully updated {config.CLEANED_CUSTOM_CSV_PATH.name} ({len(clean_df)} clean samples)!")
                    print("[!] Next step: Run 'python src/tune.py' to update your model with the new data!")
                except Exception as e:
                    print(f"[!] Warning during auto-clean: {e}")


if __name__ == '__main__':
    collector = ASLDatasetCollector()
    collector.run()
