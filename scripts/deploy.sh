#!/bin/bash
set -Eeuo pipefail
cd ~/agentdiffusion

echo '=== Step 1: Install Python deps ==='
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
.venv/bin/pip install einops scipy numpy omegaconf tqdm matplotlib pytest pandas coloredlogs

echo '=== Step 2: Install ABIDES ==='
.venv/bin/pip install vendor/abides/abides-core vendor/abides/abides-markets

echo '=== Step 3: Patch ABIDES pomegranate compat ==='
SITE=$(.venv/bin/python3 -c 'import site; print(site.getsitepackages()[0])')
cat > ${SITE}/abides_markets/models/order_size_model.py << 'PATCH_EOF'
import numpy as np
_WEIGHTS = np.array([0.2, 0.7, 0.06, 0.004, 0.0329, 0.001, 0.0006, 0.0004, 0.0005, 0.0003, 0.0003])
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()
_NORMAL_MEANS = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
class OrderSizeModel:
    def __init__(self): pass
    def sample(self, random_state):
        c = random_state.choice(len(_WEIGHTS), p=_WEIGHTS)
        if c == 0:
            val = random_state.lognormal(mean=2.9, sigma=1.2)
        else:
            val = random_state.normal(loc=_NORMAL_MEANS[c-1], scale=0.15)
        return round(max(1, val))
PATCH_EOF

echo '=== Step 4: Patch ABIDES pandas compat ==='
sed -i 's/int(pd.to_datetime(date).to_datetime64())/pd.to_datetime(date).value/' ${SITE}/abides_markets/configs/rmsc04.py

echo '=== Step 5: Smoke test ==='
.venv/bin/python3 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from agentdiffusion.utils.config import load_config
cfg = load_config()
print('Config OK:', cfg.agent.raw_dim, cfg.model.d_model)

import torch
from agentdiffusion.models.autoencoder import AgentAutoencoder
ae = AgentAutoencoder(128, 16)
print('AE OK:', ae(torch.randn(4,128))['recon'].shape)

from agentdiffusion.models.agent_dit import AgentDiT
m = AgentDiT(128,16,d_model=64,depth=2,heads=4,patch_size=4,num_market_tokens=8,local_window_size=4)
print('DiT OK:', m(torch.randn(1,16,16,16), torch.tensor([5]), torch.randn(1,32)).shape)
print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs')
PYEOF

echo '=== Step 6: Generate ABIDES data (2000 sims) ==='
.venv/bin/python3 << 'PYEOF'
import sys, logging; sys.path.insert(0, '.'); logging.disable(logging.INFO)
from agentdiffusion.data.abides_generator import generate_abides_dataset
generate_abides_dataset(output_dir='data/abides_real', num_simulations=2000, seed_start=0, end_time='11:00:00', num_snapshots=20)
PYEOF

echo '=== Step 7: Train AE ==='
.venv/bin/python3 -m agentdiffusion.train.train_ae \
    --config configs/train/stage1_ae.yaml \
    data.data_dir=data/abides_real patch.grid_h=34 patch.grid_w=33 \
    train.total_steps=5000 train.batch_size=1024 \
    train.log_every=100 train.save_every=2500 \
    data.num_workers=0 data.pin_memory=false \
    output_dir=outputs/stage1_ae_real

echo '=== Step 8: Train Diffusion ==='
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
