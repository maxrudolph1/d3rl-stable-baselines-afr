#!/bin/bash
#SBATCH --job-name=cardpol-atari
#SBATCH --output=slurm_jobs/BIG_PRETRAIN/job_%j/job_%j.out
#SBATCH --error=slurm_jobs/BIG_PRETRAIN/job_%j/job_%j.err
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

python -m afr.train_with_pretrained_encoder --data-config=afr/data_configs/seaquest_post.yaml --log-dir=artifacts/offline_rl --seed=4 --encoder-weights=null --group=BIG_PRETRAIN --freeze-encoder=True --eval-interval=500000
