import time
import os
import numpy as np
import copy
import itertools


def from_sweep_config_to_config(sweep_config):
    sweep_keys = [key for key in sweep_config.keys() if isinstance(sweep_config[key], list)]
    keys = [sweep_config[key] for key in sweep_keys]

    # Find all combinations of the lists in keys
    combinations = list(itertools.product(*keys))
    for combination in combinations:
        new_sweep_config = copy.deepcopy(sweep_config)
        for key, value in zip(sweep_keys, combination):
            new_sweep_config[key] = value
        yield new_sweep_config
        
def main(dry_run: bool = False, sweep_configs: list = None, script: str = 'main.py'):

    dict_to_args = lambda x: ' '.join([f'{key}={value}' for key, value in x.items()])

    configs = []
    for config in sweep_configs:
        configs.extend(from_sweep_config_to_config(config))

    TEMPLATE="""#!/bin/bash
#SBATCH --job-name=cardpol-atari
#SBATCH --output=slurm_jobs/{group}/job_%j/job_%j.out
#SBATCH --error=slurm_jobs/{group}/job_%j/job_%j.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=48:00:00

source /scratch/cluster/mrudolph/miniconda3/etc/profile.d/conda.sh
conda activate rlzoo
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

echo $MUJOCO_GL
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_JOB_NODELIST
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

{script} {cli_args}
"""
    import subprocess

    path = "launch_scripts/mll/temp_submission.slurm"  # You may want to modify this value or get it from config


    # Open file and append a line
    for config in configs:
        
        group_key = [key for key in config.keys() if 'group' in key][0]
        cli_args = dict_to_args(config)
        SLURM_SCRIPT = TEMPLATE.format(script=script, cli_args=cli_args, group=config[group_key])
        


        # Submit the file with sbatch
        if not dry_run:
            with open(path, "w") as f:
                f.write(SLURM_SCRIPT)
            time.sleep(1)
            result = subprocess.run(["sbatch", path], capture_output=True, text=True)
            jid = int(result.stdout.split("Submitted batch job ")[-1])
            print("sbatch output:", result.stdout)
            print("sbatch error:", result.stderr)
            
            import os
            os.makedirs(f"slurm_jobs/{config[group_key]}/job_{jid}", exist_ok=True)
            with open(f"slurm_jobs/{config[group_key]}/job_{jid}/submission.sh", "w") as f:
                f.write(SLURM_SCRIPT)
        else:
            print('DRY RUN:')
            print(f'python {script} {cli_args}')
            print('--------------------------------')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()


    ########################## TRAIN SETTINGS #################################
    # data_configs = [
    #     'configs/data/mll/cheetah_clean.yaml'
    # ]
    # exp_name = 'cheetah_clean'
    # group = 'CLEAN_CHEETAH_EXPERT_01_28_2026'
    # script = 'main.py'
    # # sweep_configs = [
    # #     {
    # #         'config': 'configs/acroaf.yaml',
    # #         'data_config': data_configs,
    # #         'seed': 0,
    # #         'model.K': 10,
    # #         'exp_name': exp_name,
    # #         'project_name': 'cardpol',
    # #         'group': group,
    # #         'train.epochs': 4,
    # #         'env.domain': 'cheetah',
    # #     },
    # sweep_configs = [
    #     {
    #         'config': 'configs/vip.yaml',
    #         'data_config': data_configs,
    #         'seed': 0,
    #         'exp_name': exp_name,
    #         'project_name': 'cardpol',
    #         'group': group,
    #         'train.epochs': 4,
    #         'env.domain': 'cheetah',
    #     },
    #     {
    #         'config': 'configs/ae.yaml',
    #         'data_config': data_configs,
    #         'seed': 0,
    #         'exp_name': exp_name,
    #         'project_name': 'cardpol',
    #         'group': group,
    #         'train.epochs': 4,
    #         'env.domain': 'cheetah',
    #     },
    #     {
    #         'config': 'configs/curl.yaml',
    #         'data_config': data_configs,
    #         'seed': 0,
    #         'exp_name': exp_name,
    #         'project_name': 'cardpol',
    #         'group': group,
    #         'train.epochs': 4,
    #         'env.domain': 'cheetah',
    #     },
    # ]

    ########################## EVAL SETTINGS #################################

    script = 'python -m afr.train_with_pretrained_encoder'
    
    # weights = ['artifacts/BreakoutNoFrameskip-v4/20260204_223530/encoder_final.pt', 'null']
    weights = ['artifacts/BreakoutNoFrameskip-v4/20260204_223530/encoder_final.pt']
    seeds = [0, 1, 2, 3, 4]
    freeze_encoders = [False]
    sweep_configs = [
        {
            '--data-config' : 'afr/data_configs/breakout_post.yaml',
            '--log-dir' : 'artifacts/offline_rl',
            '--seed' : seeds,
            # '--encoder-weights' : weights,
            '--group' : 'BREAKOUT_POST_EXPERT_02_04_2026',   
            '--eval-interval': 500000,
        }
    ]
        
    main(args.dry_run, sweep_configs, script)