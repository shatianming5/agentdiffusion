# Video DiT for Agent-Based Simulation

## Design Decision Summary

- **K=8 condition frames + N=32 generation frames** (40 total)
- **Factorized spatiotemporal attention**: spatial(81 tokens) → temporal(40 tokens)
- **Concat conditioning**: [8 clean + 32 noisy] frames processed together
- **Reuse frozen AE**: raw [36,36,128] → latent [36,36,16] → patches [81, d_model]

## Architecture

### Data Flow
```
Raw agent grid [T, 36, 36, 128]
  → AE encode (frozen) → [T, 36, 36, 16]
  → Patchify (patch_size=4) → [T, 81, d_model]
  → Video DiT (40 frames, spatial+temporal attn, v-prediction)
  → Unpatchify → [32, 36, 36, 16]
  → AE decode (frozen) → [32, 36, 36, 128]
```

### Video DiT Block
```
Per layer:
1. Spatial Self-Attention (81 tokens per frame) + adaLN-Zero(t)
2. Temporal Self-Attention (40 tokens per spatial position) + adaLN-Zero(t)
3. FFN + adaLN-Zero(t)
```

### Training
- Input: 40 consecutive agent grid frames from ABIDES
- First 8 frames clean (condition), last 32 frames noised
- Loss: v-prediction MSE on the 32 generated frames only
- Cosine noise schedule, DDIM 50-step inference

### Inference
- Condition on 8 real frames, generate 32 frames in one pass
- Sliding window: use last 8 generated as next condition for infinite rollout

### Scale
- Spatial tokens per frame: 81 (9×9 patches)
- Total tokens: 3240 (40×81)
- d_model: 512, depth: 12, heads: 8
- ~300M parameters, ~8-12GB VRAM (batch=4)

## Files to Create
- `agentdiffusion/models/video_dit.py` — VideoDiTBlock + VideoDiT
- `agentdiffusion/data/video_dataset.py` — 40-frame sequence dataset
- `agentdiffusion/train/train_video_dit.py` — training script
- `scripts/run_video_dit.sh` — training + eval pipeline
- `configs/model/video_dit.yaml` — model config
- `configs/train/stage_video_dit.yaml` — training config
