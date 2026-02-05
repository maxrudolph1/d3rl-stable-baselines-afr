#!/usr/bin/env python3
"""
CARDPOL: Contrastive Actor Recognition for Diverse POLicies

Main entry point for encoder pre-training using CARDPOL loss.

Usage:
    python -m afr_scripts.pretrain_encoder --data-config path/to/data_config.yaml

The config YAML must contain:
    environment: name of the environment (e.g. "QbertNoFrameskip-v4")
    data: list of {path: "...", label: 0} (or list of path strings; label defaults to index)
    validation_data: same structure for validation files

See pretrain_data_config_example.yaml for an example config.
"""

import argparse
import os
import sys
from datetime import datetime

import torch

import d3rlpy
from d3rlpy.preprocessing import PixelObservationScaler, ClipRewardScaler

from afr_scripts.config import load_data_config
from afr_scripts.extended_dataset import CombinedMDPDataset
from afr_scripts.losses import cardpol_loss
from afr_scripts.pretrainer import EncoderPretrainConfig, EncoderPretrainer
from afr_scripts.utils import make_atari_env


def main():
    parser = argparse.ArgumentParser(
        description="CARDPOL encoder pre-training. Pass a data config YAML with --data-config."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to YAML config with keys: environment, data (list of path/label), validation_data.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="encoder_pretrain_logs",
        help="Directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10000,
        help="Number of pre-training steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    args = parser.parse_args()

    # Load data config
    data_config = load_data_config(args.data_config)
    data_paths = data_config.data_paths
    val_data_paths = data_config.validation_data_paths
    labels = [str(l) for l in data_config.data_labels]
    val_labels = [str(l) for l in data_config.validation_data_labels]
    env_id = data_config.environment

    print(f"Environment: {env_id}")
    print(f"Training data paths: {len(data_paths)}")
    print(f"Validation data paths: {len(val_data_paths)}")

    # Create environment for action space info
    env = make_atari_env(env_id)

    # Load datasets
    print("\nLoading datasets...")
    datasets = []
    val_datasets = []

    for data_path in data_paths:
        if os.path.exists(data_path):
            data = torch.load(data_path, weights_only=False)
            dataset = d3rlpy.dataset.MDPDataset(
                observations=data['obs'],
                actions=data['action'],
                rewards=data['reward'],
                terminals=data['done'],
                action_space=d3rlpy.constants.ActionSpace.DISCRETE,
                action_size=env.action_space.n,
            )
            datasets.append(dataset)
            print(f"  Loaded {data_path}")
        else:
            print(f"  Warning: {data_path} not found, skipping...")

    for data_path in val_data_paths:
        if os.path.exists(data_path):
            data = torch.load(data_path, weights_only=False)
            dataset = d3rlpy.dataset.MDPDataset(
                observations=data['obs'],
                actions=data['action'],
                rewards=data['reward'],
                terminals=data['done'],
                action_space=d3rlpy.constants.ActionSpace.DISCRETE,
                action_size=env.action_space.n,
            )
            val_datasets.append(dataset)
            print(f"  Loaded {data_path}")
        else:
            print(f"  Warning: {data_path} not found, skipping...")

    if len(datasets) == 0:
        print("No datasets found! Exiting.")
        sys.exit(1)

    # Create combined dataset
    train_labels = labels[:len(datasets)]
    combined = CombinedMDPDataset(
        datasets=datasets,
        names=train_labels,
    )
    print(f"\n{combined}\n")

    # Create CQL model
    print("Creating CQL model...")
    cql = d3rlpy.algos.DiscreteCQLConfig(
        observation_scaler=PixelObservationScaler(),
        reward_scaler=ClipRewardScaler(-1.0, 1.0),
        compile_graph=False,
    ).create(device=args.device)

    # Build the model with one of the datasets
    cql.build_with_dataset(datasets[0])
    print("CQL model built.")
    
    # Save the model
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{args.log_dir}/{env_id}/{time_stamp}/"
    os.makedirs(log_dir, exist_ok=True)
    # cql.save(log_dir)
    print(f"Model saved to {log_dir}")

    # Create pre-trainer config
    config = EncoderPretrainConfig(
        learning_rate=1e-4,
        classifier_learning_rate=1e-4,
        batch_size=args.batch_size,
        trajectory_length=10,
        n_steps=args.n_steps,
        num_sources=len(datasets),
        classifier_hidden_sizes=[256, 128],
        classifier_combine_mode="concat",
        log_interval=100,
        save_interval=2000,
        log_dir=log_dir,
        device=args.device,
        # Validation settings
        val_interval=500,
        val_batch_size=64,
        val_n_batches=10,
        # Wandb settings
        use_wandb=not args.no_wandb,
        wandb_project="cardpol_atari_pretrain",
        wandb_run_name=f"cardpol_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        wandb_tags=["cardpol", "encoder_pretrain", env_id],
    )

    # Create validation dataset
    if val_datasets:
        val_combined = CombinedMDPDataset(
            datasets=val_datasets,
            names=val_labels[:len(val_datasets)],
        )
        print(f"Validation dataset: {val_combined}\n")
    else:
        val_combined = None
        print("Validation dataset: none (no validation paths or files not found)\n")

    # Create pre-trainer
    pretrainer = EncoderPretrainer(
        cql=cql,
        combined_dataset=combined,
        config=config,
        loss_fn=cardpol_loss,
        val_combined_dataset=val_combined,
    )

    print(f"\nEncoder architecture:")
    print(pretrainer.get_encoder(0))
    print(f"\nEncoder feature size: {pretrainer.get_encoder_feature_size()}")

    print(f"\nClassifier architecture:")
    print(pretrainer.classifier)
    print(f"Number of sources: {config.num_sources}")
    print(f"Combine mode: {config.classifier_combine_mode}")

    # Run pre-training
    print("\n" + "=" * 50)
    print("Starting pre-training...")
    print("=" * 50 + "\n")

    pretrainer.pretrain()

    # Save final encoder weights
    final_weights_path = os.path.join(log_dir, "encoder_final.pt")
    pretrainer.save_encoder_weights(final_weights_path)

    print("\n" + "=" * 50)
    print("Pre-training complete!")
    print(f"Final weights saved to: {final_weights_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
