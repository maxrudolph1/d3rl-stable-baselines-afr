"""
AFR Scripts - Actor-Free Representation learning scripts.

This package provides tools for encoder pre-training using CARDPOL
(Contrastive Actor Recognition for Diverse POLicies).
"""

from afr.networks import CARDPOLClassifier
from afr.config import DataConfig, load_data_config
from afr.extended_dataset import CombinedMDPDataset, TrajectoryBatchWithSource
from afr.losses import cardpol_loss, normalize_pixel_obs
from afr.pretrainer import (
    EncoderPretrainConfig,
    EncoderPretrainer,
    load_pretrained_encoder_to_cql,
)
from afr.utils import make_atari_env

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
