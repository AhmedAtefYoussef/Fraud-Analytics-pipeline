import os
import sys
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Prevent GUI window crashes in headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    fbeta_score,
    brier_score_loss,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

# Import and dynamically register the stacking ensemble classes for pickle compatibility
from src.models.train_model import StackingEnsemble, PyTorchTabularClassifier, TabularResNet, ResNetBlock  # type: ignore[attr-defined] # fmt: skip

setattr(sys.modules["__main__"], "StackingEnsemble", StackingEnsemble)
setattr(sys.modules["__main__"], "PyTorchTabularClassifier", PyTorchTabularClassifier)
setattr(sys.modules["__main__"], "TabularResNet", TabularResNet)
setattr(sys.modules["__main__"], "ResNetBlock", ResNetBlock)

from src.utils.logger import logger
from src.utils.config import config


def run_data_leakage_check(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> bool:
    """Validator/QA routine asserting absolute independence of training and testing data splits."""
    logger.info("Starting Programmatic Data Leakage and Integrity Audit...")

    # 1. Row intersection check
    train_hashes = pd.util.hash_pandas_object(X_train, index=False)
    test_hashes = pd.util.hash_pandas_object(X_test, index=False)

    intersect = np.intersect1d(train_hashes, test_hashes)
    overlap_count = len(intersect)

    if overlap_count > 0:
        logger.error(
            f"[LEAKAGE DETECTED]: {overlap_count} raw records overlap between train and test splits!"
        )
        return False

    # 2. Split sizing check
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("[INTEGRITY ERROR]: Split sizes are invalid (zero rows detected).")
        return False

    # 3. Label consistency
    if y_train.isna().any() or y_test.isna().any():
        logger.error("[INTEGRITY ERROR]: Target labels contain missing values.")
        return False

    logger.info(
        "Data Leakage Audit passed: Zero record overlaps, split structures validated successfully."
    )
    return True


def run_evaluation() -> None:
    """Main execution routine for Model evaluation, calibration reporting, and explainability."""
    logger.info("--- PHASE 5: MODEL EVALUATION, CALIBRATION & INTERPRETABILITY ---")

    processed_dir = config.get("data.processed_dir", "data/processed")
    models_dir = config.get("data.models_dir", "models")
    reports_dir = config.get("data.reports_dir", "reports/figures")

    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load data
    X_train_df = pd.read_csv(os.path.join(processed_dir, "X_train_final.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train_final.csv")).iloc[:, 0]

    X_test_df = pd.read_csv(os.path.join(processed_dir, "X_test_final.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test_final.csv")).iloc[:, 0]

    # 2. Leakage check
    audit_passed = run_data_leakage_check(X_train_df, X_test_df, y_train, y_test)
    if not audit_passed:
        raise ValueError(
            "Model evaluation halted due to data leakage / integrity failure."
        )

    # 3. Load model
    model_path = os.path.join(models_dir, "stacking_ensemble.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Stacking Ensemble model not found at {model_path}")
    model = joblib.load(model_path)

    # 4. Predict
    X_test = X_test_df.values
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    # 5. Calculate metrics
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    # F2 score: beta=2 balances precision and recall favoring recall (finding the fraud is crucial)
    f2 = fbeta_score(y_test, preds, beta=2.0)
    brier = brier_score_loss(y_test, probs)
    kappa = cohen_kappa_score(y_test, preds)

    logger.info("=== TEST SET PERFORMANCE METRICS ===")
    logger.info(f" - ROC-AUC:      {roc_auc:.5f}")
    logger.info(f" - PR-AUC:       {pr_auc:.5f}")
    logger.info(f" - F-beta (F2):  {f2:.5f} (Favoring Recall)")
    logger.info(f" - Brier Score:  {brier:.5f} (Lower = Better Calibration)")
    logger.info(f" - Cohen's Kappa:{kappa:.5f}")

    logger.info(
        "\nClassification Report:\n" + classification_report(y_test, preds, digits=5)
    )

    # Save Confusion Matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Approved", "Fraud"],
        yticklabels=["Approved", "Fraud"],
    )
    plt.title("Confusion Matrix - Fraud Detection Stacking Ensemble")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {cm_path}")

    # 6. Global Model Interpretability (SHAP)
    # Since Stacking ensemble is complex, we run SHAP on its primary tree component (LightGBM)
    logger.info(
        "Generating Global SHAP Interpretability explanations using LightGBM base component..."
    )
    try:
        explainer = shap.TreeExplainer(model.lgb)
        shap_values = explainer(X_test_df)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.tight_layout()
        shap_path = os.path.join(reports_dir, "shap_summary.png")
        plt.savefig(shap_path, dpi=150)
        plt.close()
        logger.info(f"Saved SHAP summary plot to: {shap_path}")
    except Exception as e:
        logger.warning(
            f"Could not calculate Tree SHAP due to: {e}. Skipping SHAP summary generation."
        )

    # 7. Local Model Interpretability (LIME)
    logger.info(
        "Generating Local LIME explanation for high-risk Fraudulent transaction..."
    )
    try:
        # Find a fraudulent row in the test set
        fraud_indices = np.where(y_test == 1)[0]
        if len(fraud_indices) > 0:
            target_idx = fraud_indices[0]

            explainer = LimeTabularExplainer(
                training_data=X_train_df.values,
                feature_names=X_train_df.columns.tolist(),
                class_names=["Approved", "Fraud"],
                mode="classification",
                random_state=42,
            )

            # Explain instance prediction
            exp = explainer.explain_instance(
                data_row=X_test[target_idx],
                predict_fn=model.predict_proba,
                num_features=8,
            )

            lime_path = os.path.join(reports_dir, "lime_explanation.html")
            exp.save_to_file(lime_path)
            logger.info(
                f"Saved interactive LIME transaction explanation to: {lime_path}"
            )
        else:
            logger.info(
                "No positive class (Fraud) transactions found in test set to explain with LIME."
            )
    except Exception as e:
        logger.warning(f"Could not calculate local LIME explanation due to: {e}")

    logger.info("Evaluation pipeline execution successfully completed.")


if __name__ == "__main__":
    run_evaluation()
