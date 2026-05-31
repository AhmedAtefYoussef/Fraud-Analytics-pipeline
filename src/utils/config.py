import os
import yaml
from typing import Any, Dict


class Config:
    """Configuration loader class that parses project yaml files."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        """Loads and parses the YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found at: {self.config_path}"
            )

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                if not config_data:
                    raise ValueError(
                        f"Configuration file at {self.config_path} is empty."
                    )
                return config_data
        except Exception as e:
            raise RuntimeError(f"Failed to parse YAML configuration: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieves a nested value from the configuration using a dot-separated path.

        Example:
            config.get("data.kaggle_dataset_slug")
        """
        keys = key_path.split(".")
        current = self._config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @property
    def raw_dict(self) -> Dict[str, Any]:
        """Returns the raw dictionary of configurations."""
        return self._config


# Global configuration instance
try:
    config = Config()
except Exception:
    # Fallback to absolute paths or specific locations if executing from subdirectories
    # In some runtimes, pwd might be different. We search upwards.
    path = "config/config.yaml"
    for _ in range(3):
        if os.path.exists(path):
            config = Config(path)
            break
        path = os.path.join("..", path)
    else:
        # Defaults if config file absolutely cannot be found
        config = None
