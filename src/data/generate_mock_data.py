import os
import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.utils.config import config


def generate_mock_dataset(
    output_dir: str = "data/raw", num_records: int = 5000, fraud_ratio: float = 0.01
) -> None:
    """Generates a realistic synthetic credit card fraud dataset for pipeline execution verification.

    Args:
        output_dir: Path where the mock creditcard.csv will be saved.
        num_records: Number of transaction records to generate.
        fraud_ratio: Proportion of fraud (Class=1) records to inject.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "creditcard.csv")

    if os.path.exists(target_path):
        logger.info(
            f"Dataset already exists at: {target_path}. Skipping mock generation."
        )
        return

    logger.info(
        f"Generating {num_records} synthetic transactions with fraud ratio {fraud_ratio:.2%}..."
    )

    np.random.seed(42)

    # 1. Class: 0 = Normal, 1 = Fraud
    num_fraud = int(num_records * fraud_ratio)
    num_normal = num_records - num_fraud

    classes = [0] * num_normal + [1] * num_fraud

    # 2. V1 - V28 features
    # Fraud transactions usually have different distributions on certain V features (e.g. V14, V17 are lower)
    v_data = {}
    for i in range(1, 29):
        col_name = f"V{i}"
        if i in [12, 14, 17]:
            # Distinct distribution for fraud
            normal_vals = np.random.normal(loc=0.0, scale=1.0, size=num_normal)
            fraud_vals = np.random.normal(loc=-3.0, scale=1.5, size=num_fraud)
        else:
            normal_vals = np.random.normal(loc=0.0, scale=0.8, size=num_normal)
            fraud_vals = np.random.normal(loc=0.0, scale=1.2, size=num_fraud)

        v_data[col_name] = np.concatenate([normal_vals, fraud_vals])

    # 3. Time: cumulative seconds
    time_normal = np.random.randint(
        0, 86400 * 2, size=num_normal
    )  # 2 days of transactions
    time_fraud = np.random.randint(0, 86400 * 2, size=num_fraud)
    times = np.concatenate([time_normal, time_fraud])

    # 4. Amount: highly right-skewed
    amount_normal = np.random.exponential(scale=50.0, size=num_normal)
    amount_fraud = np.random.exponential(scale=150.0, size=num_fraud)
    amounts = np.concatenate([amount_normal, amount_fraud])

    df = pd.DataFrame(v_data)
    df["Time"] = times
    df["Amount"] = amounts
    df["Class"] = classes

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(target_path, index=False)
    logger.info(
        f"Successfully exported synthetic dataset of size {df.shape} to: {target_path}"
    )


if __name__ == "__main__":
    generate_mock_dataset()
