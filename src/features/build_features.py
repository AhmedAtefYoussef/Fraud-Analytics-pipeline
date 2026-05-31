import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTETomek

from src.utils.logger import logger
from src.utils.config import config

class FeatureEngineer:
    """Safe, leakage-free feature engineering transformer fit strictly on training set

    and applied to other splits.
    """
    
    def __init__(self, target_encoding_smoothing: float = 10.0, kpca_components: int = 3):
        self.target_encoding_smoothing = target_encoding_smoothing
        self.kpca_components = kpca_components
        
        # Transformers to fit
        self.pca = PCA(n_components=kpca_components)
        self.global_target_mean = 0.0
        self.hour_target_map = {}
        self.fitted = False

    def _bin_time(self, X: pd.DataFrame) -> pd.Series:
        """Bins continuous Time feature into 24 hour categories."""
        if "Time" in X.columns:
            # Assuming Time is in seconds from start
            hours = (X["Time"] // 3600) % 24
            return hours.astype(int)
        # Fallback if Time not present
        return pd.Series(0, index=X.index)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "FeatureEngineer":
        """Fits PCA and target encoder smoothly using cross-fitting on X_train to prevent leakage.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            self
        """
        logger.info("Fitting Feature Engineer on training partition...")
        
        # 1. Compute global target mean
        self.global_target_mean = y_train.mean()
        
        # 2. Out-of-fold Target Encoding for binned hour
        hours = self._bin_time(X_train)
        
        # We perform out-of-fold mapping to build safe encoder mapping
        # Smooth formula: (sum(y) + smoothing * global_mean) / (count(y) + smoothing)
        temp_df = pd.DataFrame({"hour": hours, "target": y_train})
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_encoded = pd.Series(index=temp_df.index, dtype=float)
        
        for train_idx, val_idx in skf.split(temp_df, y_train):
            fold_train = temp_df.iloc[train_idx]
            fold_val = temp_df.iloc[val_idx]
            
            # Compute stats on fold_train
            stats = fold_train.groupby("hour")["target"].agg(["sum", "count"])
            smooth_map = (stats["sum"] + self.target_encoding_smoothing * self.global_target_mean) / (stats["count"] + self.target_encoding_smoothing)
            
            # Map to fold_val
            oof_encoded.iloc[val_idx] = fold_val["hour"].map(smooth_map).fillna(self.global_target_mean)

        # Store global mapping on all train data for future transform() application
        global_stats = temp_df.groupby("hour")["target"].agg(["sum", "count"])
        self.hour_target_map = (
            (global_stats["sum"] + self.target_encoding_smoothing * self.global_target_mean) /
            (global_stats["count"] + self.target_encoding_smoothing)
        ).to_dict()
        
        # 3. Fit PCA on V1-V28 columns (highly informative PCA features)
        v_cols = [c for c in X_train.columns if c.startswith("V") and len(c) <= 3]
        if v_cols:
            logger.info(f"Fitting PCA compression on {len(v_cols)} V features...")
            self.pca.fit(X_train[v_cols])
            
        self.fitted = True
        logger.info("Feature engineering components successfully fit.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms features without fitting to prevent data leakage.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: Engineered DataFrame.
        """
        if not self.fitted:
            raise RuntimeError("FeatureEngineer must be fit before transform can be called.")
            
        X_out = X.copy()
        
        # 1. Custom Domain Interactions
        # V14 and V17 are known in literature to be highly predictive interactive signals for fraud
        if "V14" in X_out.columns and "V17" in X_out.columns:
            X_out["V14_V17_interaction"] = X_out["V14"] * X_out["V17"]
            
        if "V12" in X_out.columns and "V14" in X_out.columns:
            X_out["V12_V14_interaction"] = X_out["V12"] * X_out["V14"]
            
        # Amount interaction relative to V1 (first principal component)
        if "Amount" in X_out.columns and "V1" in X_out.columns:
            X_out["Amount_V1_interaction"] = X_out["Amount"] * X_out["V1"]
            
        # 2. Add Target Encoded binned hours
        hours = self._bin_time(X_out)
        X_out["Hour_Target_Encoded"] = hours.map(self.hour_target_map).fillna(self.global_target_mean)
        
        # 3. PCA Feature Compression (Low dimension space mapping)
        v_cols = [c for c in X_out.columns if c.startswith("V") and len(c) <= 3]
        if v_cols and self.pca:
            pca_feats = self.pca.transform(X_out[v_cols])
            for i in range(self.kpca_components):
                X_out[f"PCA_Compressed_{i}"] = pca_feats[:, i]
                
        # Drop redundant/raw time if cyclical is not set
        # We keep Time and Amount since downstream models can use them
        return X_out

def run_feature_engineering() -> None:
    """Executes Phase 2.3 & 2.4: Domain feature creation, encoding, and SMOTE-Tomek imbalance balancing."""
    logger.info("--- PHASE 3: ADVANCED FEATURE ENGINEERING & IMBALANCE MITIGATION ---")
    
    processed_dir = config.get("data.processed_dir", "data/processed")
    resampling_strategy = config.get("features.resampling_strategy", "smote_tomek")
    random_state = config.get("data.random_state", 42)
    
    # 1. Load preprocessed splits
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).iloc[:, 0]
    
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).iloc[:, 0]
    
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).iloc[:, 0]
    
    # 2. Fit Feature Engineer strictly on Train
    engineer = FeatureEngineer(
        target_encoding_smoothing=config.get("features.target_encoding_smoothing", 10.0),
        kpca_components=config.get("features.kernel_pca_components", 3)
    )
    engineer.fit(X_train, y_train)
    
    # 3. Transform splits safely
    logger.info("Transforming train, val, and test splits...")
    X_train_eng = engineer.transform(X_train)
    X_val_eng = engineer.transform(X_val)
    X_test_eng = engineer.transform(X_test)
    
    # 4. Class Imbalance Mitigation (strictly on Training partition)
    if resampling_strategy == "smote_tomek":
        logger.info(f"Applying SMOTE-Tomek balancing to Train features. Initial target distribution: {y_train.value_counts().to_dict()}")
        smote_tomek = SMOTETomek(random_state=random_state, n_jobs=-1)
        X_train_res, y_train_res = smote_tomek.fit_resample(X_train_eng, y_train)
        logger.info(f"Resampling complete. Balanced target distribution: {y_train_res.value_counts().to_dict()}")
    else:
        logger.info("No class imbalance resampling requested.")
        X_train_res, y_train_res = X_train_eng, y_train
        
    # 5. Export engineered & balanced splits
    logger.info("Exporting engineered features to processed data directory...")
    X_train_res.to_csv(os.path.join(processed_dir, "X_train_final.csv"), index=False)
    y_train_res.to_csv(os.path.join(processed_dir, "y_train_final.csv"), index=False)
    
    X_val_eng.to_csv(os.path.join(processed_dir, "X_val_final.csv"), index=False)
    y_val.to_csv(os.path.join(processed_dir, "y_val_final.csv"), index=False)
    
    X_test_eng.to_csv(os.path.join(processed_dir, "X_test_final.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test_final.csv"), index=False)
    
    logger.info("Feature engineering and imbalance mitigation successfully executed.")

if __name__ == "__main__":
    run_feature_engineering()
