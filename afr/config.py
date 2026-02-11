"""
Data configuration for CARDPOL encoder pre-training.

Loads YAML with OmegaConf (environment, data paths, validation_data).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from omegaconf import OmegaConf


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
    
    
@dataclass
class EncoderPretrainConfig:
    """Configuration for encoder pre-training."""

    # Training
    learning_rate: float = 1e-3
    classifier_learning_rate: float = 1e-3
    batch_size: int = 512
    trajectory_length: int = 10
    n_steps: int = 100000

    # Classifier configuration
    num_sources: int = 2
    classifier_hidden_sizes: list = None  # Default: [256, 128]
    classifier_combine_mode: str = "concat"  # 'concat', 'diff', or 'concat_diff'

    # Behavior Cloning head configuration
    use_bc_head: bool = True  # Whether to train a BC head alongside CARDPOL
    bc_learning_rate: float = 1e-3
    bc_hidden_sizes: list = None  # Default: [256, 128]
    bc_loss_weight: float = 1.0  # Weight for BC loss relative to CARDPOL loss
    num_actions: int = 4  # Number of discrete actions (required if use_bc_head=True)

    # State decoder head configuration (decodes normalized_state from representation; no grad through encoder)
    use_state_decoder: bool = False  # Whether to train a state decoder when datasets have normalized_state
    state_dim: int = None  # Dimension of normalize[d state (required if use_state_decoder=True)
    state_decoder_learning_rate: float = 1e-3
    state_decoder_hidden_sizes: list = None  # Default: [256, 128]
    state_decoder_loss_weight: float = 1.0  # Weight for state decoder MSE loss

    # State classifier configuration (predicts source_id from normalized_state; no encoder)
    use_state_classifier: bool = False  # Whether to train a state classifier when batches have normalized_states
    state_classifier_learning_rate: float = 1e-3
    state_classifier_hidden_sizes: list = None  # Default: [256, 128]
    state_classifier_loss_weight: float = 1.0  # Weight for state classifier cross-entropy loss

    # Logging
    log_interval: int = 100
    save_interval: int = 10000
    log_dir: str = "encoder_pretrain_logs"
    group: str = "default"

    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "encoder_pretrain"
    wandb_entity: str = None  # None uses default entity
    wandb_run_name: str = None  # None auto-generates name
    wandb_tags: list = None  # Optional tags for the run

    # Validation
    val_interval: int = 500  # How often to run validation (0 to disable)
    val_batch_size: int = 64  # Batch size for validation
    val_n_batches: int = 10  # Number of batches to use for validation

    # Device
    device: str = "cuda:0"

    def __post_init__(self):
        if self.classifier_hidden_sizes is None:
            self.classifier_hidden_sizes = [256, 128]
        if self.bc_hidden_sizes is None:
            self.bc_hidden_sizes = [256, 128]
        if self.wandb_tags is None:
            self.wandb_tags = []
        if self.state_decoder_hidden_sizes is None:
            self.state_decoder_hidden_sizes = [256, 128]
        if self.state_classifier_hidden_sizes is None:
            self.state_classifier_hidden_sizes = [256, 128]
        if self.use_bc_head and self.num_actions is None:
            raise ValueError("num_actions must be specified when use_bc_head=True")
        if self.use_state_decoder and self.state_dim is None:
            raise ValueError("state_dim must be specified when use_state_decoder=True")
        if self.use_state_classifier and self.state_dim is None:
            raise ValueError("state_dim must be specified when use_state_classifier=True")


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
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping (dict).")

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
