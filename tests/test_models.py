import pytest
import numpy as np
import torch
from src.models.train_model import TabularResNet, PyTorchTabularClassifier, StackingEnsemble

def test_tabular_resnet_forward():
    """Validates that the PyTorch TabularResNet forward pass yields correct shapes and runs on CPU/GPU."""
    input_dim = 10
    hidden_dim = 32
    batch_size = 8
    
    model = TabularResNet(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=1)
    
    # Random tensor
    x = torch.randn(batch_size, input_dim)
    logits = model(x)
    
    # Assert output dimensions (batch_size, 1)
    assert logits.shape == (batch_size, 1)

def test_pytorch_wrapper_fit_predict():
    """Validates scikit-learn compliance of PyTorchTabularClassifier on dummy inputs."""
    np.random.seed(42)
    X = np.random.randn(50, 8)
    y = np.random.randint(0, 2, 50)
    
    clf = PyTorchTabularClassifier(hidden_dim=16, num_blocks=1, epochs=3, batch_size=16)
    clf.fit(X, y)
    
    # Predictions
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    
    assert probs.shape == (50, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert preds.shape == (50,)
    assert set(preds).issubset({0, 1})

def test_stacking_ensemble():
    """Validates StackingEnsemble fits and predicts across trees and PyTorch models."""
    np.random.seed(42)
    X_train = np.random.randn(40, 6)
    y_train = np.random.randint(0, 2, 40)
    
    X_val = np.random.randn(20, 6)
    y_val = np.random.randint(0, 2, 20)
    
    xgb_params = {"max_depth": 2, "n_estimators": 5}
    lgb_params = {"num_leaves": 7, "n_estimators": 5}
    py_params = {"hidden_dim": 8, "epochs": 2, "batch_size": 10}
    
    ensemble = StackingEnsemble(xgb_params, lgb_params, py_params)
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    probs = ensemble.predict_proba(X_val)
    preds = ensemble.predict(X_val)
    
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert preds.shape == (20,)
