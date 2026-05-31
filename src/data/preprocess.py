import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

from src.utils.logger import logger
from src.utils.config import config


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Loads raw dataset from the specified CSV path and deduplicates it to prevent split-level leakage.

    Args:
        data_path: Path to the raw creditcard.csv file.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")

    logger.info(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")

    # Deduplicate to prevent cross-split leakage
    initial_shape = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    deduped_shape = len(df)
    if initial_shape > deduped_shape:
        logger.info(
            f"Deduplicated raw dataset: Removed {initial_shape - deduped_shape} duplicate transaction rows."
        )

    return df


def split_data(
    df: pd.DataFrame, stratify_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strictly splits the dataset into Train (70%), Val (15%), and Test (15%) partitions

    before any preprocessing is performed, preserving class distribution.

    Args:
        df: Input raw DataFrame.
        stratify_col: Column name to use for stratified splitting.

    Returns:
        Tuple: (train_df, val_df, test_df) split DataFrames.
    """
    logger.info("Executing strict stratified train/validation/test split...")

    train_ratio = config.get("split.train_ratio", 0.70)
    val_ratio = config.get("split.val_ratio", 0.15)
    test_ratio = config.get("split.test_ratio", 0.15)
    random_state = config.get("data.random_state", 42)

    assert np.isclose(
        train_ratio + val_ratio + test_ratio, 1.0
    ), "Splits must sum to 1.0"

    # Step 1: Split into Train and Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df[stratify_col] if stratify_col in df.columns else None,
    )

    # Step 2: Split Temp into Val and Test
    val_relative_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_ratio,
        random_state=random_state,
        stratify=temp_df[stratify_col] if stratify_col in temp_df.columns else None,
    )

    logger.info(
        f"Split complete. Shapes:\n"
        f" - Train: {train_df.shape} (Fraud ratio: {train_df[stratify_col].mean():.5%})\n"
        f" - Val:   {val_df.shape} (Fraud ratio: {val_df[stratify_col].mean():.5%})\n"
        f" - Test:  {test_df.shape} (Fraud ratio: {test_df[stratify_col].mean():.5%})"
    )
    return train_df, val_df, test_df


class PreprocessingPipeline:
    """Production-grade pipeline executing scaling, imputation, and anomaly detection

    safely fit ONLY on the training partition to prevent any data leakage.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(
            n_neighbors=config.get("preprocessing.knn_neighbors", 5)
        )
        self.anomaly_detector = IsolationForest(
            contamination=config.get("preprocessing.anomaly_contamination", 0.01),
            random_state=config.get("preprocessing.outlier_random_state", 42),
            n_jobs=-1,
        )
        self.fitted = False

    def fit(
        self, train_df: pd.DataFrame, target_col: str = "Class"
    ) -> "PreprocessingPipeline":
        """Fits preprocess components (scaler, imputer, anomaly detector) ONLY on train_df.

        Args:
            train_df: Training DataFrame.
            target_col: Label column name.

        Returns:
            self
        """
        logger.info("Fitting preprocessing components on the training partition...")

        # Isolate features
        X_train = train_df.drop(columns=[target_col])

        # Fit Imputer first to handle missingness safely if any exists
        logger.info("Fitting KNN Imputer on features...")
        self.imputer.fit(X_train)
        X_imputed = self.imputer.transform(X_train)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X_train.columns)

        # Fit Scaler on all numerical features
        logger.info("Fitting StandardScaler on features...")
        self.scaler.fit(X_imputed_df)

        # Fit Isolation Forest strictly on Normal (Class == 0) training samples to capture clean normal behavior
        normal_samples = train_df[train_df[target_col] == 0].drop(columns=[target_col])
        if normal_samples.empty:
            # Fallback to all samples if normal samples cannot be filtered
            normal_samples = X_train

        logger.info(
            f"Fitting Isolation Forest on {len(normal_samples)} normal transactions..."
        )
        normal_imputed = self.imputer.transform(normal_samples)
        self.anomaly_detector.fit(normal_imputed)

        self.fitted = True
        logger.info("Preprocessing components successfully fit on training data.")
        return self

    def transform(
        self, df: pd.DataFrame, target_col: str = "Class"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Transforms a split partition using already fitted transformations.

        Args:
            df: DataFrame to transform.
            target_col: Label column name.

        Returns:
            Tuple of (X_transformed, y) where X_transformed includes normal scaled features
            and a clean anomaly score.
        """
        if not self.fitted:
            raise RuntimeError("Pipeline must be fit before calling transform.")

        y = df[target_col] if target_col in df.columns else pd.Series(dtype=np.int32)
        X = df.drop(columns=[target_col]) if target_col in df.columns else df

        # Impute
        X_imputed = self.imputer.transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        # Scale features
        X_scaled = self.scaler.transform(X_imputed_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Anomaly score: decision_function returns anomaly scores (lower means more anomalous)
        anomaly_scores = self.anomaly_detector.decision_function(X_imputed)

        # Append anomaly score as a highly informative new feature
        X_scaled_df["IsolationForest_AnomalyScore"] = anomaly_scores

        # Binary prediction (-1 represents outlier, 1 represents normal in IsolationForest)
        # Convert to 1 for anomaly, 0 for normal
        anomaly_preds = self.anomaly_detector.predict(X_imputed)
        X_scaled_df["IsolationForest_IsAnomaly"] = np.where(anomaly_preds == -1, 1, 0)

        return X_scaled_df, y


def run_preprocessing() -> None:
    """Runs Phase 2 preprocessing: loading raw data, splitting, scaling, and imputing."""
    logger.info("--- PHASE 2: PREPROCESSING & ANOMALY DETECTION ---")

    raw_dir = config.get("data.raw_dir", "data/raw")
    processed_dir = config.get("data.processed_dir", "data/processed")
    target_col = config.get("split.stratify_col", "Class")

    raw_file_path = os.path.join(raw_dir, "creditcard.csv")

    # 1. Load Raw
    df = load_raw_data(raw_file_path)

    # 2. Strict Anti-Leakage Split
    train_df, val_df, test_df = split_data(df, stratify_col=target_col)

    # 3. Instantiate & Fit Pipeline ONLY on Train
    pipeline = PreprocessingPipeline()
    pipeline.fit(train_df, target_col=target_col)

    # 4. Transform all splits safely
    logger.info("Transforming splits and creating anomaly features...")
    X_train, y_train = pipeline.transform(train_df, target_col=target_col)
    X_val, y_val = pipeline.transform(val_df, target_col=target_col)
    X_test, y_test = pipeline.transform(test_df, target_col=target_col)

    # 5. Save processed files
    os.makedirs(processed_dir, exist_ok=True)
    logger.info(f"Saving processed partitions to: {processed_dir}...")

    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)

    X_val.to_csv(os.path.join(processed_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(processed_dir, "y_val.csv"), index=False)

    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    logger.info(
        "Preprocessing execution complete. Clean partitions successfully exported."
    )


if __name__ == "__main__":
    run_preprocessing()
