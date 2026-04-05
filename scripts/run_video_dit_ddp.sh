#!/bin/bash
# Launch Video DiT training with DistributedDataParallel (2 GPUs).
#
# Usage:
#   bash scripts/run_video_dit_ddp.sh
#
# Adjust CUDA_VISIBLE_DEVICES and --nproc_per_node to match your setup.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,3

torchrun --nproc_per_node=2 -m agentdiffusion.train.train_video_dit \
    --config configs/train/stage_video_dit.yaml \
    --ae-ckpt outputs/ae_norm/ae_step_10000.pt \
    data.data_dir=data/abides_video \
    model.d_model=256 model.depth=6 model.heads=4 \
    video.num_frames=20 video.num_cond_frames=4 \
    train.total_steps=20000 train.batch_size=8 \
    train.log_every=50 train.save_every=10000 \
    data.num_workers=4 data.pin_memory=true \
    output_dir=outputs/video_dit_ddp
