"""
=============================================================================
LABS 06 & 08 INTEGRATION: MODEL TRAINING, TUNING & COMPARATIVE EVALUATION
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
This module integrates:
  - Lab 01: Preprocessed dataset ingestion & scaling (Custom vs Kaggle Benchmark)
  - Lab 06: Support Vector Machine (SVM) with RBF kernel and probability estimation
  - Lab 08: K-Nearest Neighbors (KNN) classifier benchmark & k-sensitivity analysis
  - Lab 14 CCA: Multi-dataset comparative benchmarking and artifact serialization
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from src.preprocessing import load_preprocessed_data
from src.utils import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_knn_k_sensitivity,
    plot_cross_dataset_comparison
)


# =============================================================================
# 1. LAB 06 INTEGRATION: SUPPORT VECTOR MACHINE (SVM) CLASSIFIER
# =============================================================================
def train_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = config.SVM_C,
    kernel: str = config.SVM_KERNEL,
    gamma: str = config.SVM_GAMMA,
    probability: bool = config.SVM_PROBABILITY,
    random_state: int = config.RANDOM_STATE
) -> Tuple[SVC, float]:
    """
    Trains a multi-class Support Vector Machine (SVM) classifier.
    
    Lab 06 Alignment:
      - Kernel Trick: Employs the Radial Basis Function (RBF) kernel:
            K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
      - Soft Margin C Parameter: Controls margin vs error tradeoff.
      - Probability Estimation: Enables Platt scaling (probability=True).
    """
    print("\n" + "=" * 70)
    print("TRAINING LAB 06: SUPPORT VECTOR MACHINE (SVM) CLASSIFIER")
    print(f"Hyperparameters: Kernel='{kernel}', C={C}, Gamma='{gamma}', Probability={probability}")
    print("=" * 70)

    svm = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        random_state=random_state
    )

    t0 = time.time()
    svm.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"[+] SVM training completed in {train_time:.3f} seconds.")
    print(f"[+] Total Support Vectors: {len(svm.support_)} (out of {X_train.shape[0]} training points)")
    return svm, train_time


# =============================================================================
# 2. LAB 08 INTEGRATION: K-NEAREST NEIGHBORS (KNN) CLASSIFIER
# =============================================================================
def train_knn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = config.KNN_NEIGHBORS,
    metric: str = config.KNN_METRIC,
    weights: str = config.KNN_WEIGHTS
) -> Tuple[KNeighborsClassifier, float]:
    """
    Trains a K-Nearest Neighbors (KNN) classifier.
    
    Lab 08 Alignment:
      - Instance-Based Learning: Stores training exemplars.
      - Distance Calculation: Uses Euclidean distance metric.
      - Prediction Rule: Predicts mode of k-nearest neighbors weighted by inverse distance.
    """
    print("\n" + "=" * 70)
    print("TRAINING LAB 08: K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
    print(f"Hyperparameters: n_neighbors={n_neighbors}, Metric='{metric}', Weights='{weights}'")
    print("=" * 70)

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights
    )

    t0 = time.time()
    knn.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"[+] KNN training completed in {train_time:.3f} seconds.")
    return knn, train_time


# =============================================================================
# 3. ENSEMBLE CLASSIFIER: RANDOM FOREST (RF)
# =============================================================================
def train_rf_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = config.RF_ESTIMATORS,
    random_state: int = config.RF_RANDOM_STATE
) -> Tuple[RandomForestClassifier, float]:
    """
    Trains an ensemble Random Forest classifier on 63-dimensional landmark features.
    """
    print("\n" + "=" * 70)
    print("TRAINING ENSEMBLE: RANDOM FOREST (RF) CLASSIFIER")
    print(f"Hyperparameters: n_estimators={n_estimators}, random_state={random_state}")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"[+] Random Forest training completed in {train_time:.3f} seconds.")
    return rf, train_time


def evaluate_knn_k_range(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_range: list = config.KNN_K_RANGE,
    dataset_name: str = "custom"
) -> Dict[int, float]:
    """
    Lab 08 Task 7: Tests how accuracy changes with different values of K (e.g. 3, 5, 7, 9).
    """
    print("\n--- Lab 08 Task 7: K-Sensitivity Analysis ---")
    k_accuracies = {}
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, metric=config.KNN_METRIC, weights=config.KNN_WEIGHTS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        k_accuracies[k] = acc
        print(f"   k = {k:2d} -> Test Accuracy: {acc*100:.2f}%")

    # Plot k sensitivity
    suffix = f"_{dataset_name}" if dataset_name != "custom" else ""
    knn_plot_path = config.REPORTS_DIR / f"knn_k_sensitivity{suffix}.png"
    plot_knn_k_sensitivity(list(k_accuracies.keys()), list(k_accuracies.values()), knn_plot_path)
    return k_accuracies


# =============================================================================
# 3. EVALUATION AND BENCHMARKING ENGINE (Lab 14 CCA)
# =============================================================================
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Classifier"
) -> Dict[str, Any]:
    """
    Computes a full suite of performance evaluation metrics:
      - Accuracy
      - Macro and Weighted Precision
      - Macro and Weighted Recall
      - Macro and Weighted F1-Score
      - Multi-class Confusion Matrix
      - Scikit-Learn Classification Report
      - Latency benchmark (mean inference time per sample)
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - t0) / len(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    num_classes = max(len(np.unique(y_test)), int(np.max(y_test)) + 1, int(np.max(y_pred)) + 1)
    target_names = config.LABELS_36 if num_classes > 26 else config.LABELS
    clf_report = classification_report(y_test, y_pred, target_names=target_names[:num_classes], zero_division=0)

    metrics = {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "inference_latency_ms": inference_time * 1000.0,
        "confusion_matrix": cm,
        "classification_report": clf_report
    }

    print(f"\n[{model_name} EVALUATION REPORT]")
    print(f"  * Accuracy:           {acc * 100:.2f}%")
    print(f"  * Precision (Macro):  {prec_macro * 100:.2f}% (Weighted: {prec_weighted * 100:.2f}%)")
    print(f"  * Recall (Macro):     {rec_macro * 100:.2f}% (Weighted: {rec_weighted * 100:.2f}%)")
    print(f"  * F1-Score (Macro):   {f1_m * 100:.2f}% (Weighted: {f1_w * 100:.2f}%)")
    print(f"  * Avg Inference Time: {inference_time * 1000:.3f} ms / sample")
    
    return metrics


def print_side_by_side_comparison(
    metrics_svm: Dict[str, Any],
    metrics_knn: Dict[str, Any],
    metrics_rf: Optional[Dict[str, Any]] = None
) -> None:
    """Prints side-by-side comparative table of SVM vs KNN vs Random Forest."""
    if metrics_rf is not None:
        print("\n" + "=" * 105)
        print("LAB 14 COMPLEX COMPUTING ACTIVITY (CCA): SIDE-BY-SIDE 3-MODEL BENCHMARK")
        print("=" * 105)
        header = f"{'Metric':<24} | {'SVM (Lab 06 - RBF)':<24} | {'KNN (Lab 08 - k=5)':<22} | {'Random Forest (100 Trees)':<26}"
        print(header)
        print("-" * 105)

        rows = [
            ("Accuracy", f"{metrics_svm['accuracy']*100:.2f}%", f"{metrics_knn['accuracy']*100:.2f}%", f"{metrics_rf['accuracy']*100:.2f}%"),
            ("Macro Precision", f"{metrics_svm['precision_macro']*100:.2f}%", f"{metrics_knn['precision_macro']*100:.2f}%", f"{metrics_rf['precision_macro']*100:.2f}%"),
            ("Macro Recall", f"{metrics_svm['recall_macro']*100:.2f}%", f"{metrics_knn['recall_macro']*100:.2f}%", f"{metrics_rf['recall_macro']*100:.2f}%"),
            ("Macro F1-Score", f"{metrics_svm['f1_macro']*100:.2f}%", f"{metrics_knn['f1_macro']*100:.2f}%", f"{metrics_rf['f1_macro']*100:.2f}%"),
            ("Weighted F1-Score", f"{metrics_svm['f1_weighted']*100:.2f}%", f"{metrics_knn['f1_weighted']*100:.2f}%", f"{metrics_rf['f1_weighted']*100:.2f}%"),
            ("Inference Latency", f"{metrics_svm['inference_latency_ms']:.3f} ms", f"{metrics_knn['inference_latency_ms']:.3f} ms", f"{metrics_rf['inference_latency_ms']:.3f} ms")
        ]

        for label, val_svm, val_knn, val_rf in rows:
            print(f"{label:<24} | {val_svm:<24} | {val_knn:<22} | {val_rf:<26}")
        print("=" * 105)
    else:
        print("\n" + "=" * 78)
        print("LAB 14 COMPLEX COMPUTING ACTIVITY (CCA): SIDE-BY-SIDE MODEL BENCHMARK")
        print("=" * 78)
        header = f"{'Metric':<26} | {'SVM (Lab 06 - RBF)':<24} | {'KNN (Lab 08 - k=5)':<22}"
        print(header)
        print("-" * 78)

        rows = [
            ("Accuracy", f"{metrics_svm['accuracy']*100:.2f}%", f"{metrics_knn['accuracy']*100:.2f}%"),
            ("Macro Precision", f"{metrics_svm['precision_macro']*100:.2f}%", f"{metrics_knn['precision_macro']*100:.2f}%"),
            ("Macro Recall", f"{metrics_svm['recall_macro']*100:.2f}%", f"{metrics_knn['recall_macro']*100:.2f}%"),
            ("Macro F1-Score", f"{metrics_svm['f1_macro']*100:.2f}%", f"{metrics_knn['f1_macro']*100:.2f}%"),
            ("Weighted F1-Score", f"{metrics_svm['f1_weighted']*100:.2f}%", f"{metrics_knn['f1_weighted']*100:.2f}%"),
            ("Inference Latency", f"{metrics_svm['inference_latency_ms']:.3f} ms", f"{metrics_knn['inference_latency_ms']:.3f} ms")
        ]

        for label, val_svm, val_knn in rows:
            print(f"{label:<26} | {val_svm:<24} | {val_knn:<22}")
        print("=" * 78)


# =============================================================================
# 4. DATASET-SPECIFIC TRAINING & ARTIFACT PERSISTENCE
# =============================================================================
def train_dataset_models(
    dataset_name: str = "custom",
    apply_anomaly_filter: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Executes training and evaluation for a specific dataset ('custom', 'kaggle', or 'asl36').
    Trains SVM (Lab 06), KNN (Lab 08), and Random Forest (Ensemble).
    """
    if dataset_name.lower() == "kaggle":
        mode_label = "KAGGLE BENCHMARK DATASET"
    elif dataset_name.lower() == "asl36":
        mode_label = "ASL 36 CLASSES DATASET"
    else:
        mode_label = "CUSTOM LAB DATASET"

    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION FOR: {mode_label}")
    print("=" * 80)

    # Step 1: Preprocessing & Scaling (Lab 01)
    X_train, X_test, y_train, y_test, scaler = load_preprocessed_data(
        dataset_name=dataset_name,
        apply_anomaly_filter=apply_anomaly_filter
    )

    # Step 2: SVM Classifier Training (Lab 06)
    svm_model, svm_train_time = train_svm_classifier(X_train, y_train)

    # Step 3: KNN Classifier Training (Lab 08)
    knn_model, knn_train_time = train_knn_classifier(X_train, y_train)
    evaluate_knn_k_range(X_train, y_train, X_test, y_test, dataset_name=dataset_name)

    # Step 4: Random Forest Classifier Training (Ensemble)
    rf_model, rf_train_time = train_rf_classifier(X_train, y_train)

    # Step 5: Model Evaluations
    metrics_svm = evaluate_model(svm_model, X_test, y_test, model_name=f"SVM ({mode_label})")
    metrics_knn = evaluate_model(knn_model, X_test, y_test, model_name=f"KNN ({mode_label})")
    metrics_rf = evaluate_model(rf_model, X_test, y_test, model_name=f"RF ({mode_label})")

    # Side-by-side terminal comparison across all 3 models
    print_side_by_side_comparison(metrics_svm, metrics_knn, metrics_rf)

    # Output directories and paths
    if dataset_name.lower() == "kaggle":
        target_dir = config.KAGGLE_MODELS_DIR
        cm_svm_path = config.REPORTS_DIR / "confusion_matrix_svm_kaggle.png"
        cm_knn_path = config.REPORTS_DIR / "confusion_matrix_knn_kaggle.png"
        cm_rf_path = config.REPORTS_DIR / "confusion_matrix_rf_kaggle.png"
        comp_plot_path = config.REPORTS_DIR / "model_comparison_kaggle.png"
        eval_labels = config.LABELS
    elif dataset_name.lower() == "asl36":
        target_dir = config.ASL36_MODELS_DIR
        cm_svm_path = config.REPORTS_DIR / "confusion_matrix_svm_asl36.png"
        cm_knn_path = config.REPORTS_DIR / "confusion_matrix_knn_asl36.png"
        cm_rf_path = config.REPORTS_DIR / "confusion_matrix_rf_asl36.png"
        comp_plot_path = config.REPORTS_DIR / "model_comparison_asl36.png"
        eval_labels = config.LABELS_36
    else:
        target_dir = config.CUSTOM_MODELS_DIR
        cm_svm_path = config.CONFUSION_MATRIX_SVM_PATH
        cm_knn_path = config.CONFUSION_MATRIX_KNN_PATH
        cm_rf_path = config.REPORTS_DIR / "confusion_matrix_rf.png"
        comp_plot_path = config.MODEL_COMPARISON_PLOT_PATH
        eval_labels = config.LABELS

    target_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save visual confusion matrices
    plot_confusion_matrix(
        metrics_svm["confusion_matrix"],
        labels=eval_labels,
        title=f"Lab 06: SVM Confusion Matrix ({mode_label})",
        save_path=cm_svm_path
    )
    plot_confusion_matrix(
        metrics_knn["confusion_matrix"],
        labels=eval_labels,
        title=f"Lab 08: KNN Confusion Matrix ({mode_label})",
        save_path=cm_knn_path
    )
    plot_confusion_matrix(
        metrics_rf["confusion_matrix"],
        labels=eval_labels,
        title=f"Ensemble: Random Forest Confusion Matrix ({mode_label})",
        save_path=cm_rf_path
    )

    # Also update primary report figures
    plot_confusion_matrix(
        metrics_svm["confusion_matrix"],
        labels=eval_labels,
        title=f"Lab 06: SVM Confusion Matrix ({mode_label})",
        save_path=config.CONFUSION_MATRIX_SVM_PATH
    )
    plot_confusion_matrix(
        metrics_knn["confusion_matrix"],
        labels=eval_labels,
        title=f"Lab 08: KNN Confusion Matrix ({mode_label})",
        save_path=config.CONFUSION_MATRIX_KNN_PATH
    )

    # 3-Model comparison bar chart for this dataset
    comparison_dict = {
        "SVM (Lab 06)": metrics_svm,
        "KNN (Lab 08)": metrics_knn,
        "Random Forest": metrics_rf
    }
    plot_model_comparison(comparison_dict, save_path=comp_plot_path)
    plot_model_comparison(comparison_dict, save_path=config.MODEL_COMPARISON_PLOT_PATH)

    # Step 6: Save Model Artifacts
    svm_path = target_dir / "svm_model.pkl"
    knn_path = target_dir / "knn_model.pkl"
    rf_path = target_dir / "rf_model.pkl"
    scaler_path = target_dir / "scaler.pkl"

    joblib.dump(svm_model, svm_path)
    joblib.dump(knn_model, knn_path)
    joblib.dump(rf_model, rf_path)
    joblib.dump(scaler, scaler_path)

    # Save to primary top-level models directory
    joblib.dump(svm_model, config.SVM_MODEL_PATH)
    joblib.dump(knn_model, config.KNN_MODEL_PATH)
    joblib.dump(rf_model, config.RF_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)

    print(f"\n[+] Saved {mode_label} artifacts (SVM, KNN, RF, Scaler) to: {target_dir} and {config.MODELS_DIR}")
    return metrics_svm, metrics_knn, metrics_rf


def run_cross_dataset_benchmarks():
    """
    Executes end-to-end benchmarks across ALL 3 datasets with ALL 3 models:
      1. Custom Lab Dataset (Webcam with automated anomaly cleaning)
      2. Kaggle Benchmark Dataset (Clean A-Z)
      3. ASL 36 Classes Dataset (Digits 0-9 + Letters A-Z)
    Prints a 9-way comparison table and saves a multi-dataset visualization.
    """
    print("=" * 88)
    print("ASL SIGN TRANSLATOR: TRI-DATASET & 3-MODEL BENCHMARK PIPELINE (Lab 14 CCA)")
    print("DUET AI-3202 Machine Learning Practical Manual")
    print("=" * 88)

    # 1. Custom Lab Dataset (26 classes, webcam)
    custom_svm, custom_knn, custom_rf = train_dataset_models("custom", apply_anomaly_filter=True)

    # 2. Kaggle Benchmark Dataset (26 classes, clean)
    kaggle_svm, kaggle_knn, kaggle_rf = train_dataset_models("kaggle", apply_anomaly_filter=False)

    # 3. ASL 36 Classes Dataset (36 classes: 0-9 & A-Z)
    asl36_svm, asl36_knn, asl36_rf = train_dataset_models("asl36", apply_anomaly_filter=False)

    # Grand Cross-Dataset Comparison Table (9 configurations)
    print("\n" + "=" * 90)
    print("GRAND CROSS-DATASET & MULTI-MODEL EVALUATION BENCHMARK (Lab 14 CCA)")
    print("=" * 90)
    header = f"{'Configuration':<28} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}"
    print(header)
    print("-" * 90)

    benchmarks = [
        ("Custom Lab - SVM", custom_svm),
        ("Custom Lab - KNN", custom_knn),
        ("Custom Lab - Random Forest", custom_rf),
        ("Kaggle Benchmark - SVM", kaggle_svm),
        ("Kaggle Benchmark - KNN", kaggle_knn),
        ("Kaggle Benchmark - Random Forest", kaggle_rf),
        ("ASL 36 - SVM", asl36_svm),
        ("ASL 36 - KNN", asl36_knn),
        ("ASL 36 - Random Forest", asl36_rf),
    ]

    for name, m in benchmarks:
        print(f"{name:<28} | {m['accuracy']*100:>10.2f}% | {m['precision_macro']*100:>10.2f}% | {m['recall_macro']*100:>10.2f}% | {m['f1_macro']*100:>10.2f}%")

    print("=" * 90)

    # Plot Cross-Dataset Bar Chart
    benchmark_dict = {
        "Custom (SVM)": custom_svm,
        "Custom (KNN)": custom_knn,
        "Custom (RF)": custom_rf,
        "Kaggle (SVM)": kaggle_svm,
        "Kaggle (KNN)": kaggle_knn,
        "Kaggle (RF)": kaggle_rf,
        "ASL-36 (SVM)": asl36_svm,
        "ASL-36 (KNN)": asl36_knn,
        "ASL-36 (RF)": asl36_rf,
    }
    plot_cross_dataset_comparison(benchmark_dict, save_path=config.DATASET_COMPARISON_PLOT_PATH)
    print(f"[+] Tri-dataset comparison plot saved to: {config.DATASET_COMPARISON_PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL Sign Language Classifiers (DUET AI-3202)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="asl36",
        choices=["asl36", "custom", "kaggle", "all"],
        help="Dataset to train on: 'asl36' (default), 'custom', 'kaggle', or 'all'"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Execute Lab 06 & Lab 08 hyperparameter grid search on custom dataset"
    )
    args = parser.parse_args()

    if args.tune:
        from src.tune import run_hyperparameter_tuning_pipeline
        run_hyperparameter_tuning_pipeline()
    elif args.dataset == "all":
        run_cross_dataset_benchmarks()
    else:
        train_dataset_models(args.dataset)
