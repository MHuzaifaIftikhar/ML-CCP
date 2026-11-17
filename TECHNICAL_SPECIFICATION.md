# Technical Specification & Viva Study Guide
**Project:** Real-Time ASL Sign Language to English Translator (ML × AR)  
**Course:** Machine Learning (Practical) — `AI-3202` | **Lab 14:** Complex Computing Activity (CCA)  
**Institution:** Dawood University of Engineering & Technology (DUET), Karachi (Batch 2023-F)  

---

## 1. Complete Technology Stack (What Is Used & Why)

| Component | Technology / Library | Exact Role in This Project |
| :--- | :--- | :--- |
| **Language** | Python (`3.14`) | Core runtime for data collection, training, inference, and UI |
| **Computer Vision & UI** | `opencv-python` (`>=4.8.0`) | Webcam capture (`1280×720`), frame mirroring, AR HUD drawing, mouse/touch callbacks, and evaluation graph viewer |
| **Hand Landmark Tracking** | `mediapipe` (`>=0.10.0`) | Extracts **21 3D hand keypoints** `(x, y, z)` using `HandLandmarker` (`models/hand_landmarker.task`) |
| **Machine Learning** | `scikit-learn` (`>=1.3.0`) | `StandardScaler`, `SVC` (SVM), `KNeighborsClassifier` (KNN), `GridSearchCV`, `train_test_split`, and evaluation metrics |
| **Numerical & Tabular Data** | `numpy` (`>=1.24.0`), `pandas` (`>=2.0.0`) | 63-D vector math, Euclidean norms, CSV dataset management, and `.npz` split storage |
| **Model Serialization** | `joblib` (`>=1.3.0`) | Saves and loads trained `.pkl` models (`svm_model.pkl`, `knn_model.pkl`, `scaler.pkl`) |
| **Visualization** | `matplotlib` (`>=3.7.0`), `seaborn` (`>=0.12.0`) | Generates 26-class confusion matrices, GridSearch heatmaps, KNN $k$-curves, and bar charts |
| **Text-to-Speech (TTS)** | Windows `System.Speech` (`.NET` SAPI) | 100% offline speech synthesis run on a background thread via `subprocess` (`CREATE_NO_WINDOW`) |
| **AI Sentence Completion** | `google-genai` (`gemini-3.5-flash-lite`) | Optional async background sentence suggestions (loaded via `.env`; falls back offline if unavailable) |
| **Concurrency & Buffers** | `threading`, `queue.Queue`, `collections.deque`, `Counter` | Non-blocking TTS & Gemini API calls, plus 7-frame FIFO sliding-window majority voting |

---

## 2. Input Acquisition & Feature Engineering (`src/hand_tracker.py`, `src/preprocessing.py`)

### 2.1 Camera & MediaPipe Configuration (`config.py`)
* **Resolution & Frame Rate:** `1280 × 720` (16:9 HD), target `30 FPS`, horizontally flipped (`cv2.flip(frame, 1)`) for a natural mirror display.
* **MediaPipe Parameters:**
  * `MP_MAX_NUM_HANDS = 1` (tracks dominant signing hand)
  * `MP_MIN_DETECTION_CONFIDENCE = 0.50`
  * `MP_MIN_TRACKING_CONFIDENCE = 0.50`
* **Raw Output:** A list of **21 `LandmarkPoint(x, y, z)`** objects normalized to `[0.0, 1.0]` image space.

### 2.2 Geometric Normalization (63-Feature Vector)
Raw `(x, y, z)` coordinates vary with hand position on screen and distance from the camera. Every sample (both during dataset collection and live inference) undergoes two transformations:

1. **Wrist-Origin Translation (Position Invariance):**
   $$\mathbf{p}_i' = \mathbf{p}_i - \mathbf{p}_0 \quad \forall i \in \{0, 1, \dots, 20\}$$
   Landmark `0` (Wrist) becomes `(0, 0, 0)`.
2. **Palm-Size Normalization (Scale Invariance):**
   $$\text{scale} = \|\mathbf{p}_9'\|_2 = \sqrt{(x_9')^2 + (y_9')^2 + (z_9')^2}$$
   $$\mathbf{p}_i'' = \frac{\mathbf{p}_i'}{\text{scale}}$$
   Landmark `9` is the middle finger MCP knuckle. Dividing by this distance normalizes all hand sizes.
3. **Flattening:** The `21 × 3` matrix is flattened into a `float32` vector of length **`63`** (`x0, y0, z0, ..., x20, y20, z20`).

---

## 3. Dataset & Preprocessing Pipeline (Lab 01 — `src/preprocessing.py`)

### 3.1 Dataset Structure
* **Custom Lab Dataset:** `data/raw_csv/custom_landmarks.csv` — **12,026 rows × 64 columns** (`63` float features + `1` `label` column `A–Z`).
* **Kaggle Benchmark Dataset (Optional Toggle):** `data/raw_csv/kaggle_landmarks.csv` — used for cross-dataset comparison (`[D]` key).

### 3.2 Cleaning, Anomaly Detection & Scaling Steps
1. **Null & Duplicate Handling:** Numeric nulls are imputed with column medians; missing-label rows and exact duplicate rows are dropped.
2. **Class-Centroid Anomaly Filtering (`filter_dataset_anomalies`):**
   * Groups samples by letter `A–Z`.
   * Computes each class's **median centroid** $\boldsymbol{\mu}_c \in \mathbb{R}^{63}$.
   * Computes Euclidean distance $d_i = \|\mathbf{x}_i - \boldsymbol{\mu}_c\|_2$ for every sample.
   * Keeps samples with $d_i \le \text{median}(d) + 2.2 \times \text{std}(d)$.
   * **Result:** Filtered **479 transition/blurry frames (4.0%)**, retaining **11,547 clean samples**.
3. **Label Encoding:** Maps `A–Z` $\rightarrow$ integers `0–25`.
4. **Stratified Train/Test Split:** `80%` train (**9,237** samples) / `20%` test (**2,310** samples) with `random_state=42` and `stratify=y`.
5. **Z-Score Standardization (`StandardScaler`):**
   $$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
   * **Data-Leakage Rule:** `fit_transform()` is called **only on `X_train`**; `X_test` only receives `transform()`.

---

## 4. Classification Algorithms & Tuning (Labs 06 & 08 — `src/train.py`, `src/tune.py`)

### 4.1 Support Vector Machine (Lab 06)
* **Model:** `sklearn.svm.SVC`
* **Tuned Hyperparameters (`GridSearchCV`, 5-fold stratified):**
  * `kernel='rbf'` (Radial Basis Function: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$)
  * `C=20.0` (regularization penalty)
  * `gamma=0.05` (RBF kernel bandwidth)
  * `probability=True` (Platt scaling for calibrated `predict_proba()` confidence scores)
  * `random_state=42`
* **Complexity:** `1,829` support vectors out of `9,237` training points.
* **Custom Dataset Test Performance:** **99.70% Accuracy**, **99.72% Macro Precision**, **99.69% Macro Recall**, **99.70% Macro F1**, **0.212 ms/sample** inference latency. *(On the 11,120-sample tuning benchmark, tuning improved SVM from `96.54%` baseline to `99.78%`.)*

### 4.2 K-Nearest Neighbors (Lab 08)
* **Model:** `sklearn.neighbors.KNeighborsClassifier`
* **Tuned Hyperparameters:**
  * `n_neighbors=1` (tested across $k \in [1, 3, 5, 7, 9, 11]$)
  * `metric='manhattan'` ($L_1$ distance: $\sum_{j=1}^{63} |x_j - x_{ij}|$, which preserves better distance contrast in 63-D space than Euclidean $L_2$)
  * `weights='uniform'`
* **Custom Dataset Test Performance:** **98.96% Accuracy**, **99.01% Macro Precision**, **98.97% Macro Recall**, **98.98% Macro F1**, **0.143 ms/sample** inference latency.

---

## 5. Real-Time Engine, Smoothing & AR Interaction (`app.py`, `src/utils.py`)

### 5.1 Temporal Frame-Smoothing (`TemporalSmoother`)
* Stores the last **`7`** frames in a FIFO `collections.deque(maxlen=7)`.
* Ignores predictions with `confidence < 0.50` (`CONFIDENCE_THRESHOLD`).
* Uses `collections.Counter.most_common(1)` to pick the modal letter.
* Marks the prediction **`CONFIDENT` (Green)** only when the modal letter appears in at least **`3`** frames (`CONSECUTIVE_STABLE_FRAMES = 3`).

### 5.2 Thumbs-Up Mode Toggle (`is_thumbs_up`)
Geometric check (scale-invariant relative to palm size `dist(wrist 0, middle MCP 9)`):
1. Thumb tip (`4`) is strictly above thumb IP (`3`) and MCP (`2`): `y4 < y3 < y2`, vertical extension `(y2 - y4) >= 0.42 * palm_size`, and `(min(y5, y6) - y4) >= 0.28 * palm_size`.
2. All four fingers (`8, 12, 16, 20`) are well below the thumb tip (`ty >= y4 + 0.35 * palm_size`) and curled near their MCP knuckles (`dist(tip, mcp) <= 0.78 * palm_size`), preventing false triggers on `'L'`, `'D'`, or `'G'`.
3. **Debounce:** Must be held for **`0.55 seconds`**, with a **`2.0 second`** cooldown after toggling between **`READING: ON`** and **`AR POINTER MODE`**.

### 5.3 AR Fingertip Pointer & Auto-Type Dwell
* **Auto-Type Dwell (`READING: ON`):** Holding a confident sign steady for **`1.2s`** (with **`1.5s`** repeat cooldown) appends the letter to the sentence buffer.
* **AR Pointer (`AR POINTER MODE`):** Classification pauses; **Landmark `8` (Index Fingertip)** is smoothed via EMA (`0.55 * prev + 0.45 * curr`) to draw a reticle. Hovering over any UI region (`sug_0–2`, `btn_add`, `btn_space`, `btn_back`, `btn_speak`, `btn_clear`, `btn_auto`, `btn_graphs`) for **`0.80s`** fills a circular arc and triggers the action.

---

## 6. Hybrid Sentence Suggestions & Offline Speech (`src/sentence_engine.py`)

### 6.1 `SentenceSuggestionEngine`
Ranks 3 live sentence suggestions using:
1. **ASL Shorthand Map (`SHORTHAND_MAP`):** Instant expansion for abbreviations (`WTR`, `HLP`, `TY`, `THX`, `HW`, `GM`, `GA`, `GE`, `GN`, `GB`, `PLZ`, `SRY`, `NCE`, `DUET`, `ML`).
2. **Corpus Scoring:** Scores phrases across `SENTENCE_BANK` (`A–Z`), `SHORTHAND_MAP`, and `CONVERSATIONAL_PHRASES` using exact prefix match (`+80`), word-start match (`+55`), multi-word context overlap (`+25/token`), ordered subsequence match (`+32`), and `difflib.SequenceMatcher` fuzzy ratio.
3. **Optional Async Gemini API:** Loads `GEMINI_API_KEY` from `.env` (git-ignored) and calls `gemini-3.5-flash-lite` on a background daemon thread, caching responses by `(full_text, active_letter)` so webcam FPS is never blocked.

### 6.2 `SpeechEngine`
* Uses a `queue.Queue` + daemon thread invoking Windows PowerShell `System.Speech.Synthesis.SpeechSynthesizer` with `CREATE_NO_WINDOW` (`0x08000000`).
* Sets `is_speaking = True` during playback so the HUD displays the green **`SPEAKING...`** badge.

---

## 7. Quick Viva Q&A (Common Professor Questions)

* **Q: Why use 63 landmark features instead of training a CNN directly on webcam images?**  
  **A:** Raw images are sensitive to lighting, skin tone, clothing, and background clutter. MediaPipe landmarks + wrist/palm normalization isolate pure 3D hand geometry, achieving **99.70% accuracy** with **0.21 ms** inference on CPU without needing a GPU.
* **Q: Why did you fit `StandardScaler` only on `X_train` and not the whole dataset?**  
  **A:** Fitting on the full dataset leaks test-set mean and variance into training. Fitting strictly on `X_train` ensures an honest, leakage-free test evaluation.
* **Q: Why did Manhattan (`L1`) distance beat Euclidean (`L2`) distance for KNN?**  
  **A:** In high-dimensional spaces (`63` features), $L_1$ distance suffers less from distance concentration than $L_2$, preserving sharper separation between similar hand poses.
* **Q: How do you stop the translator from flickering between letters during hand movement?**  
  **A:** Two layers: (1) centroid anomaly filtering (`std=2.2`) removes transition frames from training data, and (2) `TemporalSmoother` uses a 7-frame `deque` + `Counter` majority vote requiring `>= 3` stable frames above `0.50` confidence.
* **Q: How does the AR pointer avoid typing random letters while you move your finger to select a sentence?**  
  **A:** Holding a **Thumbs-Up** (`0.55s`) pauses letter classification and switches to **AR Pointer Mode**, turning the index fingertip into a dwell-click cursor.
