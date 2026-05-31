# Enterprise-Grade Credit Card Fraud Analytics & MLOps Pipeline

This repository implements a highly secure, enterprise-grade, end-to-end Machine Learning and MLOps Pipeline designed to detect credit card fraud using the real, highly imbalanced European cardholders transaction dataset.

The system ensembles **LightGBM**, **XGBoost**, and a custom PyTorch **Tabular ResNet** using a Stacking Classifier meta-learner, optimized via a multi-objective **Optuna study** to maximize Precision-Recall AUC (PR-AUC) while minimizing inference latency.

---

## 🚀 Key Features

* **Strict Anti-Leakage Guardrails**: Deduplicates the raw dataset and splits data into Train (70%), Val (15%), and Test (15%) partitions *before* applying any preprocessing, scaling, or oversampling.
* **State-of-the-Art Modeling**:
  * Tree models: Custom LightGBM and XGBoost classifiers.
  * Tabular Deep Learning: A PyTorch Tabular ResNet model featuring residual connections, batch normalization, and dropout regularization.
  * Ensembling: A Stacking Classifier ensembling tree models and PyTorch deep net predictions using a calibrated Logistic Regression meta-learner.
* **Imbalance Mitigation**: Fits SMOTE-Tomek resampling *exclusively on the training split* to resolve the severe class imbalance (0.172% fraud).
* **Multi-Objective Hyperparameter Optimization**: Optuna study running a TPE sampler to jointly maximize average precision score (PR-AUC) and minimize inference latency.
* **Explainability (SHAP & LIME)**: Includes global feature importance summaries via Tree SHAP and local interactive HTML transaction reports via LIME.
* **Zero-Defect CI/CD Infrastructure**: Includes comprehensive PyTest unit testing, Black formatting checks, Flake8 lints, MyPy static type checking, multi-stage Docker builds, and automated GitHub Actions workflows.

---

## 📊 Final Performance Results

Evaluated on the raw, un-sampled held-out test split (42,559 transactions, 71 fraud):

* **ROC-AUC**: `0.96109`
* **PR-AUC**: `0.80623` (highly robust under extreme imbalance)
* **F-beta (F2)**: `0.77485` (Recall-favoring balance)
* **Brier Score**: `0.00049` (exceptional probability calibration)
* **Cohen's Kappa**: `0.82144`
* **Fraud Classification Precision**: `91.38%` (minimizing false alarms; correct over 9 times out of 10!)

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.10+** installed.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Pipelines
Configurations are stored in `config/config.yaml`. Modify parameters (such as splitting ratios, resampling strategies, or Optuna trials) directly in this file.

---

## 🔁 Pipeline Execution Step-by-Step

Execute the pipeline phases sequentially using python modules:

### Phase 1: Ingest Data
Downloads the real European cardholder transactions dataset from the Hugging Face CDN (falling back to Kaggle API if credentials are present):
```bash
python -m src.data.ingest
```

### Phase 2: Split & Preprocess
Performs split-before-transform, scales continuous features, and extracts Isolation Forest anomaly features:
```bash
python -m src.data.preprocess
```

### Phase 3: Engineer Features & Resample
Applies out-of-fold target encoding, polynomial interactions, PCA components, and SMOTE-Tomek balancing:
```bash
python -m src.features.build_features
```

### Phase 4: Tune & Train Stacked Ensemble
Runs the multi-objective Optuna hyperparameter sweep, fits the final optimal stacking meta-model, and serializes it:
```bash
python -m src.models.train_model
```

### Phase 5: Leakage Audit & Evaluation
Audits splits for zero row overlap, reports classification metrics, and exports global SHAP plots and LIME reports under `reports/figures/`:
```bash
python -m src.models.evaluate
```

---

## 🧪 Unit Testing

Run the full PyTest unit verification suite:
```bash
python -m pytest tests/ -v
```

---

## 🐳 Productionization (Docker & Orchestration)

### Run with Docker Compose
We configure composing volumes to mount host-side data, models, logs, and report directories dynamically:
```bash
docker-compose up --build
```

### Build Docker Image Manually
```bash
docker build -t fraud-pipeline:latest .
```
