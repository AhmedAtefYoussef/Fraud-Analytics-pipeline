import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List

# Import and dynamically register the stacking ensemble classes for pickle compatibility
from src.models.train_model import (
    StackingEnsemble,
    PyTorchTabularClassifier,
    TabularResNet,
    ResNetBlock,
)

sys.modules["__main__"].StackingEnsemble = StackingEnsemble
sys.modules["__main__"].PyTorchTabularClassifier = PyTorchTabularClassifier
sys.modules["__main__"].TabularResNet = TabularResNet
sys.modules["__main__"].ResNetBlock = ResNetBlock

from src.utils.logger import logger
from src.utils.config import config


class InferencePipeline:
    """Production Inference Pipeline to score single transactions or batch data."""

    def __init__(self, model_path: str = "models/stacking_ensemble.joblib"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Loads the serialized ensembled model artifact."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at path: {self.model_path}"
            )
        logger.info(
            f"Loading trained Stacking Ensemble model from {self.model_path}..."
        )
        return joblib.load(self.model_path)

    def predict_record(
        self, record: Union[Dict[str, Any], pd.DataFrame]
    ) -> Dict[str, Any]:
        """Scores a single transaction record.

        Args:
            record: Dictionary of feature key-values, or a single row DataFrame.

        Returns:
            Dict: Classification prediction label and probability score.
        """
        if isinstance(record, dict):
            df = pd.DataFrame([record])
        else:
            df = record

        # Run model scoring
        probabilities = self.model.predict_proba(df.values)
        fraud_probability = float(probabilities[0, 1])
        prediction = int(fraud_probability >= 0.5)

        return {
            "fraud_probability": round(fraud_probability, 6),
            "prediction": prediction,
            "decision": "Flagged (High Risk)" if prediction == 1 else "Approved",
        }

    def predict_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Scores a batch of transaction records.

        Args:
            batch_df: DataFrame containing the transaction records.

        Returns:
            pd.DataFrame: DataFrame containing original values appended with prediction columns.
        """
        logger.info(f"Scoring batch of {len(batch_df)} transactions...")

        # Make predictions
        probabilities = self.model.predict_proba(batch_df.values)
        fraud_probabilities = probabilities[:, 1]
        predictions = np.where(fraud_probabilities >= 0.5, 1, 0)

        out_df = batch_df.copy()
        out_df["Fraud_Probability"] = fraud_probabilities
        out_df["Prediction"] = predictions
        out_df["Decision"] = np.where(
            predictions == 1, "Flagged (High Risk)", "Approved"
        )

        logger.info("Batch scoring completed.")
        return out_df


def run_inference_demo() -> None:
    """Executes a simple scoring demo of our inference pipeline."""
    logger.info("--- STARTING INFERENCE PIPELINE DEMO ---")

    processed_dir = config.get("data.processed_dir", "data/processed")
    test_path = os.path.join(processed_dir, "X_test_final.csv")

    if not os.path.exists(test_path):
        logger.error(
            f"Engineered test file not found at: {test_path}. Run features script first."
        )
        return

    # Load a few samples from test set to score
    test_data = pd.read_csv(test_path).head(5)

    pipeline = InferencePipeline()

    # 1. Test single record
    single_record = test_data.iloc[0].to_dict()
    logger.info(f"Scoring single record: {single_record}")
    result = pipeline.predict_record(single_record)
    logger.info(f"Scoring result: {result}")

    # 2. Test batch scoring
    logger.info("Scoring a batch of 5 records...")
    batch_results = pipeline.predict_batch(test_data)
    logger.info(f"\n{batch_results[['Fraud_Probability', 'Prediction', 'Decision']]}")


if __name__ == "__main__":
    run_inference_demo()
