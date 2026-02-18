#!/usr/bin/env python3
"""
Encoder pre-training: CARDPOL, VQVAE, or CURL.

Main entry point for encoder pre-training. Use 'method' in config to select:
- cardpol: contrastive source prediction (default)
- vqvae: vector-quantized reconstruction
- curl: contrastive unsupervised representations (InfoNCE)

Usage:
    # CARDPOL (default):
    python -m afr.pretrain_encoder data_config=path/to/data_config.yaml

    # VQVAE:
    python -m afr.pretrain_encoder data_config=path/to/data_config.yaml method=vqvae

    # CURL:
    python -m afr.pretrain_encoder data_config=path/to/data_config.yaml method=curl

The data config YAML must contain:
    environment: name of the environment (e.g. "QbertNoFrameskip-v4")
    data: list of {path: "...", label: 0} (or list of path strings; label defaults to index)
    validation_data: same structure for validation files
"""

from omegaconf import OmegaConf
import os
import sys
from datetime import datetime

import torch

import d3rlpy
from d3rlpy.preprocessing import PixelObservationScaler, ClipRewardScaler

from afr.config import load_data_config, EncoderPretrainConfig
from afr.extended_dataset import CombinedMDPDataset
from afr.pretrainer import EncoderPretrainer
from afr.utils import make_atari_env
from afr.networks import get_encoder_factory


def main():
    
    args = OmegaConf.from_cli()
    data_config_path = args.data_config
    del args.data_config
    
    # Load data config
    data_config = load_data_config(data_config_path)
    data_paths = data_config.data_paths
    val_data_paths = data_config.validation_data_paths
    labels = [str(l) for l in data_config.data_labels]
    val_labels = [str(l) for l in data_config.validation_data_labels]
    env_id = data_config.environment
    num_actions = data_config.num_actions
    normalized_state_dim = getattr(
        data_config, "normalized_state_dim", getattr(data_config, "state_dim", None)
    )
    # print(OmegaConf.to_yaml(config))

    # Load data config
    # data_config = load_data_config(data_config_path)
    # data_paths = data_config.data_paths
    # val_data_paths = data_config.validation_data_paths
    # labels = [str(l) for l in data_config.data_labels]
    # val_labels = [str(l) for l in data_config.validation_data_labels]
    # env_id = data_config.environment

    print(f"Environment: {env_id}")
    print(f"Training data paths: {len(data_paths)}")
    print(f"Validation data paths: {len(val_data_paths)}")

    # Create environment for action space info
    env = make_atari_env(env_id)

    # Load datasets
    print("\nLoading datasets...")
    datasets = []
    val_datasets = []
    train_states = []
    train_normalized_states = []
    val_states = []
    val_normalized_states = []

    for data_path in data_paths:
        if os.path.exists(data_path):
            data = torch.load(data_path, weights_only=False)
            if 'state' not in data or 'normalized_state' not in data:
                raise ValueError(
                    f"Dataset {data_path} is missing 'state' or 'normalized_state' keys. "
                    "All datasets must have both state and normalized_state features."
                )
            dataset = d3rlpy.dataset.MDPDataset(
                observations=data['obs'],
                actions=data['action'],
                rewards=data['reward'],
                terminals=data['done'],
                action_space=d3rlpy.constants.ActionSpace.DISCRETE,
                action_size=env.action_space.n,
            )
            datasets.append(dataset)
            train_states.append(data['state'])
            train_normalized_states.append(data['normalized_state'])
            print(f"  Loaded {data_path}")
        else:
            print(f"  Warning: {data_path} not found, skipping...")

    for data_path in val_data_paths:
        if os.path.exists(data_path):
            data = torch.load(data_path, weights_only=False)
            if 'state' not in data or 'normalized_state' not in data:
                raise ValueError(
                    f"Dataset {data_path} is missing 'state' or 'normalized_state' keys. "
                    "All datasets must have both state and normalized_state features."
                )
            dataset = d3rlpy.dataset.MDPDataset(
                observations=data['obs'],
                actions=data['action'],
                rewards=data['reward'],
                terminals=data['done'],
                action_space=d3rlpy.constants.ActionSpace.DISCRETE,
                action_size=env.action_space.n,
            )
            val_datasets.append(dataset)
            val_states.append(data['state'])
            val_normalized_states.append(data['normalized_state'])
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
        states=train_states,
        normalized_states=train_normalized_states,
        names=train_labels,
    )
    print(f"\n{combined}\n")


    
    # Save the model
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{args.log_dir}/{args.group}/{env_id}/{time_stamp}/"
    del args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    # cql.save(log_dir)
    print(f"Model saved to {log_dir}")


    
    # Create pre-trainer config
    config = EncoderPretrainConfig(
        **args,
        num_sources=len(datasets),
        log_dir=log_dir,
        wandb_project="encoder_pretrain",
        normalized_state_dim=normalized_state_dim,
        num_actions=num_actions,
    )
    # Set wandb name/tags from resolved method (after use_vqvae compat in __post_init__)
    config.wandb_run_name = f"{config.method}_{config.group}_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.wandb_tags = [config.method, "encoder_pretrain", env_id]
    
    # Create CQL model
    print("Creating CQL model...")
    
    encoder_factory = get_encoder_factory(output_size=64)
    
    cql = d3rlpy.algos.DiscreteCQLConfig(
        observation_scaler=PixelObservationScaler(),
        reward_scaler=ClipRewardScaler(-1.0, 1.0),
        compile_graph=False,
        encoder_factory=encoder_factory,
    ).create(device=config.device)

    # Build the model with one of the datasets
    cql.build_with_dataset(datasets[0])
     
    print("CQL model built.")
    
    # Create validation dataset
    if val_datasets:
        val_combined = CombinedMDPDataset(
            datasets=val_datasets,
            states=val_states,
            normalized_states=val_normalized_states,
            names=val_labels[:len(val_datasets)],
        )
        print(f"Validation dataset: {val_combined}\n")
    else:
        val_combined = None
        print("Validation dataset: none (no validation paths or files not found)\n")

    # Create pre-trainer (loss_fn=None lets pretrainer select from config.method)
    pretrainer = EncoderPretrainer(
        cql=cql,
        combined_dataset=combined,
        config=config,
        loss_fn=None,
        val_combined_dataset=val_combined,
    )

    print(f"\nEncoder architecture:")
    print(pretrainer.get_encoder(0))
    print(f"\nEncoder feature size: {pretrainer.get_encoder_feature_size()}")

    if config.method == "vqvae":
        print(f"\nVQVAE (quantizer + decoder):")
        print(pretrainer.aux_module)
        print(f"num_embeddings={config.vqvae_num_embeddings}, commitment_cost={config.vqvae_commitment_cost}")
    elif config.method == "curl":
        print(f"\nCURL (encoder-only): pad={config.curl_pad}, temperature={config.curl_temperature}")
    else:
        print(f"\nCARDPOL classifier:")
        print(pretrainer.aux_module)
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
