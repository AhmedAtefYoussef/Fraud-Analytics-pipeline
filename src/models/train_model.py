import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

from src.utils.logger import logger
from src.utils.config import config

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using PyTorch device: {DEVICE}")


# PyTorch Tabular ResNet model implementation
class ResNetBlock(nn.Module):
    """A residual block for tabular data containing linear layers, batchnorm, dropout, and residual sum."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.act2(x)
        return x


class TabularResNet(nn.Module):
    """Multi-layer Tabular ResNet designed to learn highly non-linear fraud signals."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        self.act_input = nn.SiLU()

        self.blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, 1)  # Single node for binary logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.bn_input(x)
        x = self.act_input(x)

        for block in self.blocks:
            x = block(x)

        logits = self.output_layer(x)
        return logits


# Scikit-learn compatible PyTorch Wrapper
class PyTorchTabularClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible PyTorch Tabular ResNet classifier wrapper."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
        lr: float = 0.001,
        batch_size: int = 512,
        epochs: int = 20,
        patience: int = 5,
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.classes_ = np.array([0, 1])

        self.model: Optional[TabularResNet] = None
        self.input_dim_ = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "PyTorchTabularClassifier":
        # Handle formats
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        self.input_dim_ = X_arr.shape[1]
        self.model = TabularResNet(
            input_dim=self.input_dim_,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        ).to(DEVICE)

        # Prepare datasets
        train_dataset = TensorDataset(torch.tensor(X_arr), torch.tensor(y_arr))
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        # Validation setup
        best_loss = float("inf")
        best_weights = None
        patience_counter = 0

        eval_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
            eval_loader = DataLoader(
                eval_dataset, batch_size=self.batch_size, shuffle=False
            )

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(batch_X).squeeze(1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_X)

            train_loss /= len(X_arr)

            # Validation Step
            if eval_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in eval_loader:
                        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                        logits = self.model(batch_X).squeeze(1)
                        loss = criterion(logits, batch_y)
                        val_loss += loss.item() * len(batch_X)
                val_loss /= len(X_val)

                # Check for Early Stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping at epoch {epoch}. Restoring best weights."
                        )
                        self.model.load_state_dict(
                            {k: v.to(DEVICE) for k, v in best_weights.items()}
                        )
                        break

        if eval_set is not None and best_weights is not None:
            self.model.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_weights.items()}
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.tensor(X_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(DEVICE)
                logits = self.model(batch_X).squeeze(1)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p)

        probs = np.array(probs)
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return np.where(prob >= 0.5, 1, 0)


# Stacking Ensemble model definition
class StackingEnsemble:
    """Ensemble model combining tree classifiers and tabular neural network via stacking."""

    def __init__(
        self,
        xgb_params: Dict[str, Any],
        lgb_params: Dict[str, Any],
        pytorch_params: Dict[str, Any],
    ):
        self.xgb = XGBClassifier(**xgb_params, random_state=42, n_jobs=-1)
        self.lgb = LGBMClassifier(**lgb_params, random_state=42, n_jobs=-1, verbose=-1)
        self.pytorch = PyTorchTabularClassifier(**pytorch_params)

        self.meta_learner = LogisticRegression(C=1.0, penalty="l2", random_state=42)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "StackingEnsemble":
        logger.info("Fitting XGBoost base classifier...")
        self.xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        logger.info("Fitting LightGBM base classifier...")
        self.lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

        logger.info("Fitting PyTorch Tabular ResNet base classifier...")
        self.pytorch.fit(X_train, y_train, eval_set=(X_val, y_val))

        # Build out-of-fold predictions on validation set for stacking
        logger.info("Fitting logistic meta-learner stacking classifier...")
        p_xgb = self.xgb.predict_proba(X_val)[:, 1]
        p_lgb = self.lgb.predict_proba(X_val)[:, 1]
        p_torch = self.pytorch.predict_proba(X_val)[:, 1]

        meta_features = np.column_stack([p_xgb, p_lgb, p_torch])
        self.meta_learner.fit(meta_features, y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        p_torch = self.pytorch.predict_proba(X)[:, 1]

        meta_features = np.column_stack([p_xgb, p_lgb, p_torch])
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return np.where(p >= 0.5, 1, 0)


# Multi-Objective Optuna study to optimize Precision-Recall AUC and Inference Latency
def run_tuning(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Runs a multi-objective hyperparameter optimization using Optuna (PR-AUC vs Latency)."""
    logger.info("Starting Multi-Objective Optuna study (TPE Sampler)...")

    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        # Suggest params
        xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.2, log=True)
        xgb_depth = trial.suggest_int("xgb_depth", 3, 7)
        xgb_n_est = trial.suggest_int("xgb_n_est", 50, 150)

        lgb_lr = trial.suggest_float("lgb_lr", 0.01, 0.2, log=True)
        lgb_leaves = trial.suggest_int("lgb_leaves", 15, 63)

        py_hidden = trial.suggest_categorical("py_hidden", [64, 128, 256])
        py_dropout = trial.suggest_float("py_dropout", 0.05, 0.4)

        # Build ensemble
        xgb_params = {
            "learning_rate": xgb_lr,
            "max_depth": xgb_depth,
            "n_estimators": xgb_n_est,
        }
        lgb_params = {
            "learning_rate": lgb_lr,
            "num_leaves": lgb_leaves,
            "n_estimators": 100,
        }
        pytorch_params = {
            "hidden_dim": py_hidden,
            "dropout": py_dropout,
            "epochs": 5,
            "batch_size": 1024,
        }

        ensemble = StackingEnsemble(xgb_params, lgb_params, pytorch_params)

        # Train
        ensemble.fit(X_train, y_train, X_val, y_val)

        # Evaluate PR-AUC
        preds_proba = ensemble.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, preds_proba)

        # Evaluate Latency (time to predict a batch of 1000 items)
        latencies = []
        batch_sample = X_val[:1000] if len(X_val) >= 1000 else X_val
        for _ in range(3):  # Take average of 3 runs
            start = time.perf_counter()
            _ = ensemble.predict_proba(batch_sample)
            latencies.append(time.perf_counter() - start)

        avg_latency = np.mean(latencies)

        return pr_auc, avg_latency

    n_trials = config.get("models.optuna.n_trials", 15)
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(
            seed=config.get("models.optuna.random_state", 42)
        ),
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info("Optuna study complete. Extracting best trials...")
    # Find trial with the best trade-off (highest PR-AUC)
    best_trials = study.best_trials
    logger.info(f"Number of Pareto-optimal trials: {len(best_trials)}")

    # Pick trial maximizing PR-AUC
    best_trial = max(best_trials, key=lambda t: t.values[0])
    logger.info(
        f"Selected Optimal Trial values: PR-AUC = {best_trial.values[0]:.4f}, Latency = {best_trial.values[1]:.4f}s"
    )

    # Format and return params
    params = best_trial.params
    xgb_params = {
        "learning_rate": params["xgb_lr"],
        "max_depth": params["xgb_depth"],
        "n_estimators": params["xgb_n_est"],
    }
    lgb_params = {
        "learning_rate": params["lgb_lr"],
        "num_leaves": params["lgb_leaves"],
        "n_estimators": 100,
    }
    pytorch_params = {
        "hidden_dim": params["py_hidden"],
        "dropout": params["py_dropout"],
        "epochs": 20,
        "batch_size": 512,
    }

    return xgb_params, lgb_params, pytorch_params


def run_training() -> None:
    """Executes Phase 3 training pipeline: loading engineered datasets, tuning hyperparams, ensembling."""
    logger.info("--- PHASE 4: STATE-OF-THE-ART MODEL TRAINING ---")

    processed_dir = config.get("data.processed_dir", "data/processed")
    models_dir = config.get("data.models_dir", "models")

    # 1. Load engineered sets
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train_final.csv")).values
    y_train = pd.read_csv(
        os.path.join(processed_dir, "y_train_final.csv")
    ).values.squeeze()

    X_val = pd.read_csv(os.path.join(processed_dir, "X_val_final.csv")).values
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val_final.csv")).values.squeeze()

    # 2. Run Hyperparameter Tuning
    xgb_params, lgb_params, py_params = run_tuning(X_train, y_train, X_val, y_val)

    # 3. Train final optimal Stacking Ensemble
    logger.info("Training final optimal Stacking Ensemble on complete training data...")
    optimal_ensemble = StackingEnsemble(xgb_params, lgb_params, py_params)
    optimal_ensemble.fit(X_train, y_train, X_val, y_val)

    # 4. Save model artifact
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "stacking_ensemble.joblib")
    logger.info(f"Saving final Stacking Ensemble to {model_path}...")

    # Since PyTorch state inside joblib can be tricky due to DEVICE binding,
    # we compile predictions easily. joblib handles it perfectly, but we make sure pytorch model is kept.
    joblib.dump(optimal_ensemble, model_path)
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    run_training()
