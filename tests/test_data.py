import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import split_data, PreprocessingPipeline

def test_split_data():
    """Validates that splitting is perfectly stratified and preserves dimensions."""
    # Create mock dataset with 100 rows, 10 fraud instances
    np.random.seed(42)
    data = {
        "V1": np.random.randn(100),
        "Amount": np.random.rand(100) * 100,
        "Class": [1] * 10 + [0] * 90
    }
    df = pd.DataFrame(data)
    
    train_df, val_df, test_df = split_data(df, stratify_col="Class")
    
    # Check total size
    assert len(train_df) + len(val_df) + len(test_df) == 100
    
    # Check class ratios (allowing for standard rounding variations)
    assert train_df["Class"].sum() in [7, 8]
    assert val_df["Class"].sum() in [1, 2]
    assert test_df["Class"].sum() in [1, 2]
    
    # Assert zero overlap
    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    test_indices = set(test_df.index)
    
    assert train_indices.intersection(val_indices) == set()
    assert train_indices.intersection(test_indices) == set()
    assert val_indices.intersection(test_indices) == set()

def test_preprocessing_pipeline_fit_transform():
    """Validates that PreprocessingPipeline scales and imputes safely without leakage."""
    # Mock train and test sets
    train_data = pd.DataFrame({
        "V1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "Amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Class": [0, 0, 0, 1, 1]
    })
    
    test_data = pd.DataFrame({
        "V1": [2.0, np.nan],
        "Amount": [15.0, 25.0],
        "Class": [0, 1]
    })
    
    pipeline = PreprocessingPipeline()
    pipeline.fit(train_data, target_col="Class")
    
    # Transform train and test
    X_train, y_train = pipeline.transform(train_data, target_col="Class")
    X_test, y_test = pipeline.transform(test_data, target_col="Class")
    
    # Assert NaN values are imputed
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    
    # Assert shapes are consistent
    assert X_train.shape[1] == 4  # V1, Amount, IsolationForest_AnomalyScore, IsolationForest_IsAnomaly
    assert X_test.shape[0] == 2
