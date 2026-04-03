#!/bin/bash
set -Eeuo pipefail
cd ~/agentdiffusion
export PYTHONPATH=$PWD:${PYTHONPATH:-}

echo '=== Step 6: Generate ABIDES data (2000 sims) ==='
.venv/bin/python3 -c "
import logging; logging.disable(logging.INFO)
from agentdiffusion.data.abides_generator import generate_abides_dataset
generate_abides_dataset(output_dir='data/abides_real', num_simulations=2000, seed_start=0, end_time='11:00:00', num_snapshots=20)
"

echo '=== Step 7: Train AE (5000 steps) ==='
.venv/bin/python3 -m agentdiffusion.train.train_ae \
    --config configs/train/stage1_ae.yaml \
    data.data_dir=data/abides_real patch.grid_h=34 patch.grid_w=33 \
    train.total_steps=5000 train.batch_size=1024 \
    train.log_every=100 train.save_every=2500 \
    data.num_workers=0 data.pin_memory=false \
    output_dir=outputs/stage1_ae_real

echo '=== Step 8: Train Diffusion (5000 steps) ==='
.venv/bin/python3 -m agentdiffusion.train.train_diffusion \
    --config configs/train/stage2_diffusion.yaml \
    --ae-ckpt outputs/stage1_ae_real/ae_step_5000.pt \
    data.data_dir=data/abides_real patch.grid_h=36 patch.grid_w=36 \
    patch.patch_size=4 model.d_model=128 model.depth=4 model.heads=4 \
    model.num_market_tokens=16 model.local_window_size=4 \
    agent.latent_dim=16 diffusion.timesteps=200 \
    train.total_steps=5000 train.batch_size=4 \
    train.log_every=50 train.save_every=2500 \
    data.num_workers=0 data.pin_memory=false \
    output_dir=outputs/stage2_dit_real

echo '=== ALL DONE ==='
