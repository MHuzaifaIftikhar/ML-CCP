"""
=============================================================================
HYPERPARAMETER TUNING ENGINE (Labs 06 & 08 Alignment)
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
This module integrates:
  - Lab 06 Task 4: Hyperparameter Tuning of Support Vector Machine (SVM)
    using GridSearchCV (C, gamma, kernel) with 5-Fold Stratified Cross-Validation.
  - Lab 08 Task 4 & Task 7: Hyperparameter Tuning of K-Nearest Neighbors (KNN)
    using GridSearchCV (k, weights, metric) and k-sensitivity analysis.
  - Lab 14 CCA: Side-by-side comparative benchmarking of Baseline vs. Tuned models,
    generating publication-ready visualization heatmaps and serialized artifacts.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
from src.utils import plot_confusion_matrix


# =============================================================================
# 1. LAB 06 TASK 4: SVM HYPERPARAMETER TUNING VIA GRIDSEARCHCV
# =============================================================================
def tune_svm_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    n_jobs: int = -1
) -> Tuple[SVC, Dict[str, Any], pd.DataFrame]:
    """
    Performs systematic hyperparameter optimization for Support Vector Machine.
    
    Lab 06 Task 4 Alignment:
      - Searches over regularization parameter C (soft-margin penalty)
      - Searches over kernel functions (RBF vs Linear)
      - Searches over gamma (RBF kernel bandwidth parameter)
      - Evaluates via 5-Fold Stratified Cross-Validation
      - Fits final best estimator with Platt probability calibration
    """
    print("\n" + "=" * 78)
    print("LAB 06 TASK 4: SVM HYPERPARAMETER TUNING (GridSearchCV + 5-Fold CV)")
    print("=" * 78)

    # Grid specification
    param_grid = [
        {
            'kernel': ['rbf'],
            'C': [0.5, 1.0, 5.0, 10.0, 20.0],
            'gamma': ['scale', 'auto', 0.01, 0.05, 0.1]
        },
        {
            'kernel': ['linear'],
            'C': [0.1, 0.5, 1.0, 5.0]
        }
    ]

    total_candidates = (len(param_grid[0]['C']) * len(param_grid[0]['gamma'])) + len(param_grid[1]['C'])
    print(f"[*] Exploring {total_candidates} parameter combinations across {cv_folds}-Fold Stratified CV ({total_candidates * cv_folds} total fits)...")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Use probability=False during GridSearch for fast fitting across folds
    base_svc = SVC(probability=False, random_state=config.RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=base_svc,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - t0

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"\n[+] SVM GridSearch completed in {search_time:.2f} seconds.")
    print("[+] Optimal Hyperparameters Discovered:")
    for k, v in best_params.items():
        print(f"      - {k}: {v}")
    print(f"[+] Best 5-Fold CV Mean Accuracy: {best_cv_score * 100:.2f}%")

    # Refit final optimal model with probability=True (Platt scaling for AR HUD display)
    print("[*] Calibrating optimal SVM with Platt probability scaling (probability=True)...")
    t_refit = time.time()
    optimal_svm = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        gamma=best_params.get('gamma', 'scale'),
        probability=True,
        random_state=config.RANDOM_STATE
    )
    optimal_svm.fit(X_train, y_train)
    print(f"[+] Calibrated SVM refitted in {time.time() - t_refit:.2f} seconds.")
    print(f"[+] Support Vector Count: {len(optimal_svm.support_)} / {X_train.shape[0]}")

    return optimal_svm, best_params, cv_results_df


# =============================================================================
# 2. LAB 08 TASK 4 & 7: KNN HYPERPARAMETER TUNING VIA GRIDSEARCHCV
# =============================================================================
def tune_knn_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    n_jobs: int = -1
) -> Tuple[KNeighborsClassifier, Dict[str, Any], pd.DataFrame]:
    """
    Performs systematic hyperparameter optimization for K-Nearest Neighbors.
    
    Lab 08 Tasks 4 & 7 Alignment:
      - Searches over neighborhood size k (number of neighbors)
      - Searches over distance metrics: Euclidean (L2) vs Manhattan (L1)
      - Searches over neighbor weighting: Uniform vs Distance-weighted
      - Evaluates via 5-Fold Stratified Cross-Validation
    """
    print("\n" + "=" * 78)
    print("LAB 08 TASK 4: KNN HYPERPARAMETER TUNING (GridSearchCV + 5-Fold CV)")
    print("=" * 78)

    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    total_candidates = len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])
    print(f"[*] Exploring {total_candidates} parameter combinations across {cv_folds}-Fold Stratified CV ({total_candidates * cv_folds} total fits)...")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    base_knn = KNeighborsClassifier()

    grid_search = GridSearchCV(
        estimator=base_knn,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - t0

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"\n[+] KNN GridSearch completed in {search_time:.2f} seconds.")
    print("[+] Optimal Hyperparameters Discovered:")
    for k, v in best_params.items():
        print(f"      - {k}: {v}")
    print(f"[+] Best 5-Fold CV Mean Accuracy: {best_cv_score * 100:.2f}%")

    optimal_knn = KNeighborsClassifier(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        metric=best_params['metric']
    )
    optimal_knn.fit(X_train, y_train)

    return optimal_knn, best_params, cv_results_df


# =============================================================================
# 3. EVALUATION ENGINE & BENCHMARK COMPARISON
# =============================================================================
def evaluate_model_performance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Evaluates comprehensive metrics on test partition."""
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - t0) / len(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=config.LABELS, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "inference_latency_ms": inference_time * 1000.0,
        "confusion_matrix": cm,
        "classification_report": report
    }


def print_tuning_comparison_table(
    svm_baseline: Dict[str, Any],
    svm_tuned: Dict[str, Any],
    knn_baseline: Dict[str, Any],
    knn_tuned: Dict[str, Any]
) -> None:
    """Prints comprehensive 4-way comparison table."""
    print("\n" + "=" * 90)
    print("HYPERPARAMETER TUNING BENCHMARK: BASELINE vs TUNED MODELS (Lab 14 CCA)")
    print("=" * 90)
    header = f"{'Metric':<22} | {'SVM Baseline':<14} | {'SVM Tuned':<14} | {'KNN Baseline':<14} | {'KNN Tuned':<14}"
    print(header)
    print("-" * 90)

    rows = [
        ("Accuracy",
         f"{svm_baseline['accuracy']*100:.2f}%", f"{svm_tuned['accuracy']*100:.2f}%",
         f"{knn_baseline['accuracy']*100:.2f}%", f"{knn_tuned['accuracy']*100:.2f}%"),
        ("Macro Precision",
         f"{svm_baseline['precision_macro']*100:.2f}%", f"{svm_tuned['precision_macro']*100:.2f}%",
         f"{knn_baseline['precision_macro']*100:.2f}%", f"{knn_tuned['precision_macro']*100:.2f}%"),
        ("Macro Recall",
         f"{svm_baseline['recall_macro']*100:.2f}%", f"{svm_tuned['recall_macro']*100:.2f}%",
         f"{knn_baseline['recall_macro']*100:.2f}%", f"{knn_tuned['recall_macro']*100:.2f}%"),
        ("Macro F1-Score",
         f"{svm_baseline['f1_macro']*100:.2f}%", f"{svm_tuned['f1_macro']*100:.2f}%",
         f"{knn_baseline['f1_macro']*100:.2f}%", f"{knn_tuned['f1_macro']*100:.2f}%"),
        ("Inference Latency",
         f"{svm_baseline['inference_latency_ms']:.3f} ms", f"{svm_tuned['inference_latency_ms']:.3f} ms",
         f"{knn_baseline['inference_latency_ms']:.3f} ms", f"{knn_tuned['inference_latency_ms']:.3f} ms"),
    ]

    for label, s_b, s_t, k_b, k_t in rows:
        print(f"{label:<22} | {s_b:<14} | {s_t:<14} | {k_b:<14} | {k_t:<14}")
    print("=" * 90)


# =============================================================================
# 4. TUNING VISUALIZATIONS
# =============================================================================
def plot_svm_tuning_heatmap(
    cv_results_df: pd.DataFrame,
    save_path: Path
) -> None:
    """
    Plots a 2D heatmap showing mean CV test accuracy for RBF SVM across C and gamma.
    """
    rbf_results = cv_results_df[cv_results_df['param_kernel'] == 'rbf'].copy()
    if rbf_results.empty:
        return

    # Pivot table: rows = C, cols = gamma
    rbf_results['param_C'] = rbf_results['param_C'].astype(str)
    rbf_results['param_gamma'] = rbf_results['param_gamma'].astype(str)
    pivot = rbf_results.pivot(index='param_C', columns='param_gamma', values='mean_test_score')

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={'label': '5-Fold CV Accuracy (%)'},
        linewidths=0.8,
        linecolor='white'
    )
    plt.title("Lab 06 Task 4: SVM (RBF) Hyperparameter Tuning Heatmap (C vs. Gamma)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Kernel Coefficient (Gamma)", fontsize=11, labelpad=8)
    plt.ylabel("Regularization Parameter (C)", fontsize=11, labelpad=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] SVM tuning heatmap saved to: {save_path}")


def plot_knn_tuning_comparison(
    cv_results_df: pd.DataFrame,
    save_path: Path
) -> None:
    """
    Plots KNN CV accuracy curves across k values grouped by metric and weighting scheme.
    """
    plt.figure(figsize=(10, 6))
    
    # Filter and plot each combination of metric and weight
    for metric in ['euclidean', 'manhattan']:
        for weight in ['uniform', 'distance']:
            subset = cv_results_df[
                (cv_results_df['param_metric'] == metric) &
                (cv_results_df['param_weights'] == weight)
            ].sort_values('param_n_neighbors')
            
            linestyle = '-' if weight == 'distance' else '--'
            marker = 'o' if metric == 'euclidean' else 's'
            label = f"{metric.capitalize()} ({weight})"
            
            plt.plot(
                subset['param_n_neighbors'],
                subset['mean_test_score'] * 100,
                label=label,
                linestyle=linestyle,
                marker=marker,
                linewidth=2
            )

    plt.title("Lab 08 Task 4 & 7: KNN Hyperparameter Tuning & k-Sensitivity", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Neighbors (k)", fontsize=11)
    plt.ylabel("5-Fold CV Mean Accuracy (%)", fontsize=11)
    plt.xticks(sorted(cv_results_df['param_n_neighbors'].unique()))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='lower left')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] KNN tuning curves saved to: {save_path}")


def plot_baseline_vs_tuned_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    save_path: Path
) -> None:
    """
    Grouped bar chart comparing Baseline vs. Tuned performance for SVM and KNN.
    """
    categories = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    models = list(comparison_data.keys())
    
    x = np.arange(len(categories))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#aec7e8', '#1f77b4', '#98df8a', '#2ca02c']  # Light/Dark Blue (SVM), Light/Dark Green (KNN)

    for i, model in enumerate(models):
        values = [
            comparison_data[model]['accuracy'] * 100,
            comparison_data[model]['precision_macro'] * 100,
            comparison_data[model]['recall_macro'] * 100,
            comparison_data[model]['f1_macro'] * 100
        ]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=model, color=colors[i], edgecolor='black')
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Lab 14 CCA: Baseline vs. Tuned Classifier Performance (Custom Dataset)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='semibold')
    ax.set_ylim(85, 105)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Baseline vs. Tuned comparison chart saved to: {save_path}")


# =============================================================================
# 5. MAIN TUNING WORKFLOW PIPELINE
# =============================================================================
def run_hyperparameter_tuning_pipeline() -> Tuple[SVC, KNeighborsClassifier]:
    """
    Executes end-to-end hyperparameter tuning on the custom dataset.
    """
    print("=" * 84)
    print("STARTING ASL MODEL HYPERPARAMETER TUNING PIPELINE (Labs 06, 08, 14 CCA)")
    print("=" * 84)

    # 1. Ingest clean custom dataset
    X_train, X_test, y_train, y_test, scaler = load_preprocessed_data(
        dataset_name="custom",
        apply_anomaly_filter=False  # Cleaned CSV is already present and validated
    )

    # 2. Baseline Model Evaluations (pre-tuning performance)
    print("\n[*] Evaluating Baseline Models on Test Set...")
    baseline_svm = SVC(
        C=config.SVM_C,
        kernel=config.SVM_KERNEL,
        gamma=config.SVM_GAMMA,
        probability=True,
        random_state=config.RANDOM_STATE
    )
    baseline_svm.fit(X_train, y_train)
    metrics_svm_base = evaluate_model_performance(baseline_svm, X_test, y_test, "SVM Baseline")

    baseline_knn = KNeighborsClassifier(
        n_neighbors=config.KNN_NEIGHBORS,
        metric=config.KNN_METRIC,
        weights=config.KNN_WEIGHTS
    )
    baseline_knn.fit(X_train, y_train)
    metrics_knn_base = evaluate_model_performance(baseline_knn, X_test, y_test, "KNN Baseline")

    # 3. Systematic Hyperparameter Tuning
    tuned_svm, best_svm_params, svm_cv_df = tune_svm_hyperparameters(X_train, y_train)
    tuned_knn, best_knn_params, knn_cv_df = tune_knn_hyperparameters(X_train, y_train)

    # 4. Tuned Model Evaluations on Test Set
    metrics_svm_tuned = evaluate_model_performance(tuned_svm, X_test, y_test, "SVM Tuned")
    metrics_knn_tuned = evaluate_model_performance(tuned_knn, X_test, y_test, "KNN Tuned")

    # 5. Comparative Reporting
    print_tuning_comparison_table(metrics_svm_base, metrics_svm_tuned, metrics_knn_base, metrics_knn_tuned)

    # 6. Generate Publication-Quality Visual Reports
    reports_dir = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    svm_heatmap_path = reports_dir / "svm_gridsearch_heatmap.png"
    knn_tuning_path = reports_dir / "knn_tuning_curves.png"
    comp_chart_path = reports_dir / "baseline_vs_tuned_comparison.png"
    cm_svm_path = reports_dir / "confusion_matrix_svm_tuned.png"
    cm_knn_path = reports_dir / "confusion_matrix_knn_tuned.png"

    plot_svm_tuning_heatmap(svm_cv_df, svm_heatmap_path)
    plot_knn_tuning_comparison(knn_cv_df, knn_tuning_path)

    comparison_data = {
        "SVM (Baseline)": metrics_svm_base,
        "SVM (Tuned)": metrics_svm_tuned,
        "KNN (Baseline)": metrics_knn_base,
        "KNN (Tuned)": metrics_knn_tuned
    }
    plot_baseline_vs_tuned_comparison(comparison_data, comp_chart_path)

    plot_confusion_matrix(
        metrics_svm_tuned["confusion_matrix"],
        labels=config.LABELS,
        title=f"Lab 06: Tuned SVM Confusion Matrix (C={best_svm_params['C']}, kernel='{best_svm_params['kernel']}')",
        save_path=cm_svm_path
    )
    plot_confusion_matrix(
        metrics_knn_tuned["confusion_matrix"],
        labels=config.LABELS,
        title=f"Lab 08: Tuned KNN Confusion Matrix (k={best_knn_params['n_neighbors']}, metric='{best_knn_params['metric']}')",
        save_path=cm_knn_path
    )

    # 7. Persist Tuned Model Artifacts
    custom_models_dir = config.CUSTOM_MODELS_DIR
    custom_models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Serializing tuned model artifacts to disk...")
    # Save to custom models folder
    joblib.dump(tuned_svm, custom_models_dir / "svm_model.pkl")
    joblib.dump(tuned_knn, custom_models_dir / "knn_model.pkl")
    joblib.dump(scaler, custom_models_dir / "scaler.pkl")

    # Also update root models folder for active use in app.py
    joblib.dump(tuned_svm, config.SVM_MODEL_PATH)
    joblib.dump(tuned_knn, config.KNN_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)

    # Save summary report text file
    summary_report_path = reports_dir / "tuning_summary.txt"
    with open(summary_report_path, "w", encoding="utf-8") as f:
        f.write("ASL SIGN TRANSLATOR: HYPERPARAMETER TUNING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: Cleaned Custom Dataset ({len(X_train) + len(X_test)} total samples)\n")
        f.write(f"Training split: {len(X_train)} samples, Test split: {len(X_test)} samples\n\n")
        f.write("OPTIMAL SVM HYPERPARAMETERS (Lab 06 Task 4):\n")
        for k, v in best_svm_params.items():
            f.write(f"  - {k}: {v}\n")
        f.write(f"  Test Accuracy: {metrics_svm_tuned['accuracy']*100:.2f}% (Baseline: {metrics_svm_base['accuracy']*100:.2f}%)\n")
        f.write(f"  Test Macro F1: {metrics_svm_tuned['f1_macro']*100:.2f}% (Baseline: {metrics_svm_base['f1_macro']*100:.2f}%)\n\n")
        f.write("OPTIMAL KNN HYPERPARAMETERS (Lab 08 Task 4 & 7):\n")
        for k, v in best_knn_params.items():
            f.write(f"  - {k}: {v}\n")
        f.write(f"  Test Accuracy: {metrics_knn_tuned['accuracy']*100:.2f}% (Baseline: {metrics_knn_base['accuracy']*100:.2f}%)\n")
        f.write(f"  Test Macro F1: {metrics_knn_tuned['f1_macro']*100:.2f}% (Baseline: {metrics_knn_base['f1_macro']*100:.2f}%)\n")

    print(f"[+] Tuned models serialized successfully to {custom_models_dir} and {config.MODELS_DIR}")
    print(f"[+] Text report written to {summary_report_path}")
    print("\n" + "=" * 84)
    print("HYPERPARAMETER TUNING COMPLETE [100% SUCCESS]")
    print("=" * 84)

    return tuned_svm, tuned_knn


if __name__ == "__main__":
    run_hyperparameter_tuning_pipeline()
