import pytest
import pandas as pd
import numpy as np
from src.features.build_features import FeatureEngineer


def test_feature_engineer():
    """Validates that FeatureEngineer constructs, bins, encodes, and compresses correctly."""
    # Create mock inputs with V features, Time, and Amount
    np.random.seed(42)
    X_train = pd.DataFrame(
        {
            "V1": np.random.randn(20),
            "V2": np.random.randn(20),
            "V3": np.random.randn(20),
            "Time": np.arange(0, 72000, 3600),  # 20 hours
            "Amount": np.random.rand(20) * 100,
        }
    )
    y_train = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])

    engineer = FeatureEngineer(target_encoding_smoothing=2.0, kpca_components=2)
    engineer.fit(X_train, y_train)

    X_train_trans = engineer.transform(X_train)

    # Assert engineered features are present
    assert "Hour_Target_Encoded" in X_train_trans.columns
    assert "PCA_Compressed_0" in X_train_trans.columns
    assert "PCA_Compressed_1" in X_train_trans.columns
    assert (
        "V12_V14_interaction" not in X_train_trans.columns
    )  # V12/V14 not present in mock

    # Check shape of outputs
    assert X_train_trans.shape[0] == 20
    assert X_train_trans.shape[1] > X_train.shape[1]

    # Transform test set (with a different time)
    X_test = pd.DataFrame(
        {
            "V1": np.random.randn(5),
            "V2": np.random.randn(5),
            "V3": np.random.randn(5),
            "Time": [3600 * 2, 3600 * 5, 3600 * 10, 3600 * 15, 3600 * 24],
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    X_test_trans = engineer.transform(X_test)
    assert not X_test_trans.isna().any().any()
    assert X_test_trans.shape[0] == 5
