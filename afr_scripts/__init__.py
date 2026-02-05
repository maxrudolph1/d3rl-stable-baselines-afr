"""
AFR Scripts - Actor-Free Representation learning scripts.

This package provides tools for encoder pre-training using CARDPOL
(Contrastive Actor Recognition for Diverse POLicies).
"""

from afr_scripts.classifier import CARDPOLClassifier
from afr_scripts.config import DataConfig, load_data_config
from afr_scripts.extended_dataset import CombinedMDPDataset, TrajectoryBatchWithSource
from afr_scripts.losses import cardpol_loss, normalize_pixel_obs
from afr_scripts.pretrainer import (
    EncoderPretrainConfig,
    EncoderPretrainer,
    load_pretrained_encoder_to_cql,
)
from afr_scripts.utils import make_atari_env

__all__ = [
    # Config
    "DataConfig",
    "load_data_config",
    # Dataset
    "CombinedMDPDataset",
    "TrajectoryBatchWithSource",
    # Classifier
    "CARDPOLClassifier",
    # Losses
    "cardpol_loss",
    "normalize_pixel_obs",
    # Pretrainer
    "EncoderPretrainConfig",
    "EncoderPretrainer",
    "load_pretrained_encoder_to_cql",
    # Utils
    "make_atari_env",
]
