"""
Data configuration for CARDPOL encoder pre-training.

Provides YAML-based configuration loading for specifying data paths,
validation paths, labels, and environment name.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class DataConfig:
    """
    Data configuration loaded from a YAML file.

    Attributes:
        environment: Name of the environment (e.g. 'QbertNoFrameskip-v4').
        data_paths: List of paths to training data files.
        data_labels: List of source labels for each training path (same order as data_paths).
        validation_data_paths: List of paths to validation data files.
        validation_data_labels: List of source labels for each validation path.
    """
    environment: str
    data_paths: List[str]
    data_labels: List[Union[int, str]]
    validation_data_paths: List[str]
    validation_data_labels: List[Union[int, str]]


def load_data_config(config_path: Union[str, Path]) -> DataConfig:
    """
    Load data paths, validation paths, and environment name from a YAML config.

    Expected YAML structure:
        environment: "QbertNoFrameskip-v4"
        data:
          - path: "path/to/train_0.pth"
            label: 0
          - path: "path/to/train_1.pth"
            label: 1
        validation_data:
          - path: "path/to/val_0.pth"
            label: 0
          - path: "path/to/val_1.pth"
            label: 1

    Each entry under `data` and `validation_data` can be:
      - A dict with "path" and optional "label" (defaults to index).
      - A string (path only; label defaults to index).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        DataConfig with parsed data paths, labels, and environment.

    Raises:
        RuntimeError: If PyYAML is not installed.
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is malformed.
    """
    if not YAML_AVAILABLE:
        raise RuntimeError(
            "Loading config from YAML requires PyYAML. Install with: pip install pyyaml"
        )
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)
        
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping (dict).")

    environment = raw.get("environment")
    if environment is None:
        raise ValueError("Config must contain 'environment' (name of the environment).")
    if not isinstance(environment, str):
        raise ValueError("'environment' must be a string.")

    def parse_entries(key: str) -> Tuple[List[str], List[Union[int, str]]]:
        entries = raw.get(key)
        if entries is None:
            return [], []
        if not isinstance(entries, list):
            raise ValueError(f"'{key}' must be a list of entries.")
        paths: List[str] = []
        labels: List[Union[int, str]] = []
        for i, entry in enumerate(entries):
            if isinstance(entry, str):
                paths.append(entry)
                labels.append(i)
            elif isinstance(entry, dict):
                p = entry.get("path")
                if p is None:
                    raise ValueError(f"Entry in '{key}' must have 'path' or be a string.")
                paths.append(str(p))
                labels.append(entry.get("label", i))
            else:
                raise ValueError(f"Entry in '{key}' must be a string or a dict with 'path'.")
        return paths, labels

    data_paths, data_labels = parse_entries("data")
    validation_data_paths, validation_data_labels = parse_entries("validation_data")

    return DataConfig(
        environment=environment,
        data_paths=data_paths,
        data_labels=data_labels,
        validation_data_paths=validation_data_paths,
        validation_data_labels=validation_data_labels,
    )
