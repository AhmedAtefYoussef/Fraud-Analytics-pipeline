import os
import time
import zipfile
from typing import Optional
from src.utils.logger import logger
from src.utils.config import config

# Load .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def authenticate_kaggle() -> bool:
    """Checks and sets up Kaggle API authentication.

    Returns:
        bool: True if authentication configuration is found, False otherwise.
    """
    # Check environment variables
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        # Set them in os.environ where the kaggle package expects them
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        logger.info("Kaggle credentials detected in environment variables.")
        return True

    # Check user home folder for kaggle.json
    home_dir = os.path.expanduser("~")
    kaggle_json = os.path.join(home_dir, ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        logger.info(f"Kaggle configuration file found at: {kaggle_json}")
        return True

    logger.warning(
        "Kaggle API credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY "
        "env variables or place 'kaggle.json' in ~/.kaggle/"
    )
    return False


def download_from_mirror(url: str, output_path: str) -> bool:
    """Downloads the real creditcard.csv directly from a public mirror in chunks with progress logging."""
    try:
        import requests

        logger.info(
            f"Attempting automated direct HTTP download from Git LFS mirror: {url}"
        )

        # Start streaming the request
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 2 * 1024 * 1024  # 2MB chunks for speed

        logger.info(
            f"Streaming data... Total file size: {total_size / (1024*1024):.2f} MB"
        )

        downloaded = 0
        last_log_time = time.time()

        with open(output_path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)

                # Log progress at most once every 3 seconds to avoid spamming the log file
                current_time = time.time()
                if current_time - last_log_time >= 3.0:
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        logger.info(
                            f"Downloaded: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)"
                        )
                    else:
                        logger.info(f"Downloaded: {downloaded / (1024*1024):.1f} MB")
                    last_log_time = current_time

        # Log final progress
        logger.info(
            f"Direct stream download complete! Final size: {downloaded / (1024*1024):.2f} MB"
        )
        return True
    except Exception as e:
        logger.warning(f"Mirror download failed: {e}")
        return False


def download_dataset(
    dataset_slug: str, output_dir: str, max_retries: int = 3, backoff_factor: int = 2
) -> bool:
    """Downloads dataset from Kaggle using API, falling back to public Git LFS mirror if auth fails."""
    os.makedirs(output_dir, exist_ok=True)

    # Check if target file already exists
    csv_file = os.path.join(output_dir, "creditcard.csv")
    if os.path.exists(csv_file):
        logger.info(f"Real dataset already exists at: {csv_file}. Skipping download.")
        return True

    # Primary Method: Kaggle API
    has_auth = authenticate_kaggle()
    if has_auth:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            attempt = 1
            delay = 2
            while attempt <= max_retries:
                try:
                    logger.info(
                        f"Attempt {attempt}: Initializing Kaggle API connection..."
                    )
                    api = KaggleApi()
                    api.authenticate()

                    logger.info(
                        f"Downloading dataset '{dataset_slug}' to '{output_dir}'..."
                    )
                    api.dataset_download_files(
                        dataset_slug, path=output_dir, unzip=True
                    )

                    # Verify CSV file is created
                    if os.path.exists(csv_file):
                        logger.info(
                            "Dataset successfully downloaded and extracted from Kaggle!"
                        )
                        return True
                    else:
                        # Extract zip manually if needed
                        for file in os.listdir(output_dir):
                            if file.endswith(".zip"):
                                zip_path = os.path.join(output_dir, file)
                                logger.info(f"Extracting downloaded zip: {zip_path}")
                                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                                    zip_ref.extractall(output_dir)
                                os.remove(zip_path)

                        if os.path.exists(csv_file):
                            logger.info("Dataset manually extracted and validated!")
                            return True
                    raise FileNotFoundError(
                        "creditcard.csv was not found after download and extraction."
                    )
                except Exception as e:
                    logger.warning(f"Kaggle API attempt {attempt} failed: {e}")
                    if attempt == max_retries:
                        break
                    time.sleep(delay)
                    attempt += 1
                    delay *= backoff_factor
        except Exception as e:
            logger.warning(f"Kaggle API initialization failed: {e}")

    # Secondary Method: Direct Hugging Face CDN Mirror download
    logger.info(
        "Kaggle API unavailable or failed. Initiating fallback download of real creditcard.csv from Hugging Face CDN..."
    )
    mirror_url = "https://huggingface.co/datasets/JEFFREY-VERDIERE/Creditcard/resolve/main/creditcard.csv"
    success = download_from_mirror(mirror_url, csv_file)

    if not success:
        # Retry with standard raw URL in case LFS CDN is blocked
        raw_url = "https://raw.githubusercontent.com/chsety/Credit-Card-Fraud-Detection/master/creditcard.csv"
        logger.info("Retrying with raw GitHub URL...")
        success = download_from_mirror(raw_url, csv_file)

    return success


def run_ingestion() -> None:
    """Main execution entrypoint for data ingestion."""
    logger.info("--- PHASE 1: DATA INGESTION ---")

    dataset_slug = config.get("data.kaggle_dataset_slug", "mlg-ulb/creditcardfraud")
    raw_dir = config.get("data.raw_dir", "data/raw")

    success = download_dataset(dataset_slug=dataset_slug, output_dir=raw_dir)

    if not success:
        logger.error(
            f"\n[CRITICAL ERROR]: All download sources failed.\n"
            f"Please manually download the dataset from 'https://www.kaggle.com/datasets/{dataset_slug}' "
            f"and place 'creditcard.csv' directly inside '{os.path.abspath(raw_dir)}/' so we can proceed!"
        )
        raise RuntimeError(
            "Data ingestion failed and no download sources are accessible."
        )


if __name__ == "__main__":
    run_ingestion()
