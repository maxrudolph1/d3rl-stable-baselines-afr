#!/usr/bin/env python3
"""
Train a CQL policy using a pre-trained encoder.

This script loads pre-trained encoder weights from CARDPOL pre-training
and uses them to train a policy with CQL on a target dataset.

Usage:
    python -m afr.train_with_pretrained_encoder \
        --data-config path/to/data_config.yaml \
        --encoder-weights path/to/encoder_final.pt \
        --train-data path/to/training_data.pth

Or use the data config for training data as well:
    python -m afr.train_with_pretrained_encoder \
        --data-config path/to/data_config.yaml \
        --encoder-weights path/to/encoder_final.pt \
        --use-config-data
"""

import argparse
import os
import sys

import torch
import d3rlpy
from d3rlpy.preprocessing import PixelObservationScaler, ClipRewardScaler
from d3rlpy.logging import FileAdapterFactory

from afr.config import load_data_config
from afr.pretrainer import load_pretrained_encoder_to_cql
from afr.utils import make_atari_env
from afr.networks import get_encoder_factory


def load_dataset_from_path(data_path: str, env) -> d3rlpy.dataset.MDPDataset:
    """Load a dataset from a .pth file."""
    data = torch.load(data_path, weights_only=False)
    dataset = d3rlpy.dataset.MDPDataset(
        observations=data['obs'],
        actions=data['action'],
        rewards=data['reward'],
        terminals=data['done'],
        action_space=d3rlpy.constants.ActionSpace.DISCRETE,
        action_size=env.action_space.n,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train CQL policy with pre-trained encoder."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to YAML data config (for environment name).",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default=None,
        help="Path to pre-trained encoder weights (.pt file). If not provided, uses random weights.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data file (.pth). If not provided, uses first entry from data config.",
    )
    parser.add_argument(
        "--use-config-data",
        action="store_true",
        help="Use all data paths from config (combined into one dataset).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="artifacts/offline_rl",
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
        default=1000000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10000,
        help="Interval for saving checkpoints.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10000,
        help="Interval for running environment evaluation.",
    )
    parser.add_argument(
        "--freeze-encoder",
        type=bool,
        default=False,
        help="Freeze encoder weights during training.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable environment evaluation during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility.",
    )
    
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Project name for logging.",
    )
    args = parser.parse_args()

    # Load data config for environment name
    data_config = load_data_config(args.data_config)
    env_id = data_config.environment
    print(f"Environment: {env_id}")

    # Create environment
    env = make_atari_env(env_id)

    # Determine training data path(s)
    if args.train_data is not None:
        train_data_paths = [args.train_data]
    elif args.use_config_data:
        train_data_paths = data_config.data_paths
    else:
        # Default: use first data path from config
        if len(data_config.data_paths) == 0:
            print("Error: No training data specified. Use --train-data or --use-config-data.")
            sys.exit(1)
        train_data_paths = [data_config.data_paths[0]]

    print(f"Training data paths: {train_data_paths}")

    # Load training dataset(s)
    print("\nLoading training data...")
    datasets = []
    for data_path in train_data_paths:
        if os.path.exists(data_path):
            dataset = load_dataset_from_path(data_path, env)
            datasets.append(dataset)
            print(f"  Loaded {data_path} ({len(dataset.episodes)} episodes)")
        else:
            print(f"  Warning: {data_path} not found, skipping...")

    if len(datasets) == 0:
        print("Error: No training data loaded!")
        sys.exit(1)

    # Combine datasets if multiple
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        # Concatenate all episodes
        import numpy as np
        all_obs = np.concatenate([d.episodes[0].observations for d in datasets for _ in range(len(d.episodes))])
        # Actually we need to properly combine - let's use the first for now
        # TODO: Implement proper dataset concatenation
        print("Warning: Multiple datasets provided, using first one only for now.")
        dataset = datasets[0]

    print(f"\nDataset: {len(dataset.episodes)} episodes")
    
    

    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # Create CQL model
    print("\nCreating CQL model...")
    encoder_factory = get_encoder_factory(output_size=64)
    cql = d3rlpy.algos.DiscreteCQLConfig(
        observation_scaler=PixelObservationScaler(),
        reward_scaler=ClipRewardScaler(-1.0, 1.0),
        compile_graph=False,
        encoder_factory=encoder_factory,
    ).create(device=args.device)

    # Build the model with the dataset
    cql.build_with_dataset(dataset)
    print("CQL model built.")

    # Load pre-trained encoder weights
    if args.encoder_weights is not None and args.encoder_weights != 'null' and os.path.exists(args.encoder_weights) and args.encoder_weights != 'None':
        print(f"\nLoading pre-trained encoder from: {args.encoder_weights}")
        load_pretrained_encoder_to_cql(cql, args.encoder_weights, device=args.device)
    else:
        print("No pre-trained encoder weights provided, using random weights.")

    # Optionally freeze encoder weights
    if args.freeze_encoder:
        print("Freezing encoder weights...")
        for q_func in cql._impl._modules.q_funcs:
            for param in q_func.encoder.parameters():
                param.requires_grad = False
        for targ_q_func in cql._impl._modules.targ_q_funcs:
            for param in targ_q_func.encoder.parameters():
                param.requires_grad = False
        print("Encoder weights frozen.")

    # Set up logging
    from datetime import datetime
    import random
    import string
    # Make the random tag much less likely to collide by including more entropy
    # Use time, process id, and extra random characters
    import socket
    job_id = os.environ.get('SLURM_JOB_ID', 'no_job_id')
    rand_tag = ( ''.join(random.choices(string.ascii_letters + string.digits, k=6)) + f"_{os.getpid()}_{job_id}")
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_tag = f"{time_stamp}_{rand_tag}"
    if args.group is not None:
        log_dir = f"{args.log_dir}/{args.group}/{env_id}/"
    else:
        log_dir = f"{args.log_dir}/{env_id}/"
        
    os.makedirs(log_dir, exist_ok=True)
    logger_factory = FileAdapterFactory(log_dir)

    # Set up evaluators
    evaluators = {}
    if not args.no_eval:
        eval_env = make_atari_env(env_id)
        evaluators["environment"] = d3rlpy.metrics.EnvironmentEvaluator(
            eval_env, epsilon=0.001
        )
        print(f"Environment evaluation enabled (every {args.eval_interval} steps)")

    from omegaconf import OmegaConf

    config_save_dir = os.path.join(log_dir, unique_tag)
    os.makedirs(config_save_dir, exist_ok=True)
    config_save_path = os.path.join(config_save_dir, "config.yaml")
    OmegaConf.save(
        OmegaConf.create({"args": vars(args), "unique_tag": unique_tag}),
        config_save_path,
    )
    print(f"Saved config to: {config_save_path}")
    
    # Start training
    print("\n" + "=" * 50)
    print("Starting CQL training with pre-trained encoder...")
    print(f"  Steps: {args.n_steps}")
    print(f"  Save interval: {args.save_interval}")
    print(f"  Log dir: {log_dir}")
    print(f"  Encoder frozen: {args.freeze_encoder}")
    print("=" * 50 + "\n")
    

    cql.fit(
        dataset,
        n_steps=args.n_steps,
        save_interval=args.save_interval,
        evaluators=evaluators if evaluators else None,
        logger_adapter=logger_factory,
        experiment_name=unique_tag,
        with_timestamp=False,
    )

    # Save final model
    final_model_path = os.path.join(args.log_dir, "cql_final.d3")
    # INSERT_YOUR_CODE

    
    cql.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Run final evaluation
    if not args.no_eval:
        print("\nRunning final evaluation...")
        final_evaluator = d3rlpy.metrics.EnvironmentEvaluator(
            eval_env, epsilon=0.001, n_trials=10
        )
        mean_return = final_evaluator(cql, dataset)
        print(f"Final mean return: {mean_return}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
