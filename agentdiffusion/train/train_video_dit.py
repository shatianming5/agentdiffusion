"""Stage: Video DiT training — train factorized spatiotemporal diffusion on agent sequences.

Training pipeline:
    1. Load 40-frame sequences from AgentVideoDataset.
    2. Encode all frames through a frozen AE to get latent representations.
    3. Split into K=8 condition frames (kept clean) and N=32 generation frames.
    4. Add noise to generation frames only, at a random timestep t.
    5. The model sees [8 clean + 32 noisy] and predicts v for the 32 generation frames.
    6. Loss = MSE(v_pred, v_target) on generation frames.

v-prediction: v = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x_0
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.video_dit import build_video_dit, VideoDiT
from ..models.autoencoder import AgentAutoencoder
from ..diffusion.scheduler import NoiseScheduler
from ..data.video_dataset import AgentVideoDataset, SyntheticVideoDataset
from ..utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA (identical pattern to train_diffusion.py)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.lerp_(m_param.data, 1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# Cosine warmup + decay learning rate schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay to 0."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class VideoDiTTrainer:
    """Full training loop for the Video DiT model."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Frozen autoencoder (from Stage 1) ---
        self.ae = AgentAutoencoder(
            raw_dim=cfg.agent.raw_dim,
            latent_dim=cfg.agent.latent_dim,
        ).to(self.device)
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        # --- Video DiT model ---
        self.model = build_video_dit(cfg).to(self.device)

        # --- Noise scheduler (cosine) ---
        self.scheduler = NoiseScheduler(
            timesteps=cfg.diffusion.timesteps,
            schedule=cfg.diffusion.beta_schedule,
        ).to(self.device)

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=cfg.train.warmup_steps,
            total_steps=cfg.train.total_steps,
        )

        # --- EMA ---
        self.ema = EMA(self.model, cfg.train.ema_decay)

        self.global_step = 0
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Video-specific params
        self.num_cond_frames = cfg.video.num_cond_frames
        self.num_gen_frames = cfg.video.num_frames - cfg.video.num_cond_frames

        # Mixed precision
        self.use_amp = cfg.train.mixed_precision in ("fp16", "bf16")
        self.amp_dtype = (
            torch.bfloat16 if cfg.train.mixed_precision == "bf16" else torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16))

    def load_ae_checkpoint(self, path: str):
        """Load pretrained autoencoder weights."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.ae.load_state_dict(ckpt["model"])
        logger.info("Loaded AE from %s", path)

    def load_checkpoint(self, path: str):
        """Resume from a Video DiT checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("step", 0)
        if "ema" in ckpt:
            self.ema.shadow.load_state_dict(ckpt["ema"])
        logger.info("Resumed Video DiT from %s (step %d)", path, self.global_step)

    def build_dataloader(self) -> DataLoader:
        """Build the training DataLoader."""
        p = self.cfg.patch.patch_size
        pad_h = ((self.cfg.patch.grid_h + p - 1) // p) * p
        pad_w = ((self.cfg.patch.grid_w + p - 1) // p) * p

        total_frames = self.cfg.video.num_frames
        cond_frames = self.cfg.video.num_cond_frames

        try:
            dataset = AgentVideoDataset(
                data_dir=self.cfg.data.data_dir,
                total_frames=total_frames,
                cond_frames=cond_frames,
                pad_to=(pad_h, pad_w),
                market_cond_dim=getattr(self.cfg.model, "market_cond_dim", 32),
            )
            logger.info("Loaded %d video sequences from %s", len(dataset), self.cfg.data.data_dir)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Real data not available (%s), using synthetic data", e)
            dataset = SyntheticVideoDataset(
                num_samples=500,
                total_frames=total_frames,
                cond_frames=cond_frames,
                grid_h=pad_h,
                grid_w=pad_w,
                raw_dim=self.cfg.agent.raw_dim,
                market_cond_dim=getattr(self.cfg.model, "market_cond_dim", 32),
            )

        return DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step.

        Returns dict of scalar metrics for logging.
        """
        frames = batch["frames"].to(self.device)       # [B, T, H, W, C_raw]
        market_conds = batch["market_conds"].to(self.device)  # [B, T, cond_dim]

        B, T, H, W, C_raw = frames.shape
        K = self.num_cond_frames
        N = self.num_gen_frames

        # --- Encode all frames through frozen AE ---
        with torch.no_grad():
            flat_frames = frames.reshape(B * T, H, W, C_raw)
            latents = self.ae.encode(flat_frames)             # [B*T, H, W, d_latent]
            latents = latents.reshape(B, T, H, W, -1)        # [B, T, H, W, d_latent]

        # Split condition and generation frames
        z_cond = latents[:, :K]      # [B, K, H, W, d_latent] -- clean
        z_gen = latents[:, K:]       # [B, N, H, W, d_latent] -- to be noised

        # --- Diffusion: sample timestep and add noise to generation frames ---
        t = torch.randint(0, self.scheduler.timesteps, (B,), device=self.device)
        noise = torch.randn_like(z_gen)

        # Flatten gen frames for q_sample (needs flat batch dim)
        z_gen_flat = z_gen.reshape(B * N, H, W, -1)
        noise_flat = noise.reshape(B * N, H, W, -1)
        t_expanded = t.unsqueeze(1).expand(B, N).reshape(B * N)
        z_noisy_flat = self.scheduler.q_sample(z_gen_flat, t_expanded, noise_flat)
        z_noisy = z_noisy_flat.reshape(B, N, H, W, -1)

        # --- Forward pass ---
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            v_pred = self.model(z_cond, z_noisy, t)  # [B, N, H, W, d_latent]

            # Compute v-target for generation frames
            d_latent = z_gen.shape[-1]
            z_gen_flat2 = z_gen.reshape(B * N, H, W, d_latent)
            noise_flat2 = noise.reshape(B * N, H, W, d_latent)
            v_target_flat = self.scheduler.v_target(z_gen_flat2, noise_flat2, t_expanded)
            v_target = v_target_flat.reshape(B, N, H, W, d_latent)

            # MSE loss on generation frames only
            loss = F.mse_loss(v_pred, v_target)

        # --- Backward ---
        self.optimizer.zero_grad()
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.optimizer.step()

        self.lr_scheduler.step()
        self.ema.update(self.model)

        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train(self):
        """Full training loop."""
        loader = self.build_dataloader()
        pbar = tqdm(
            total=self.cfg.train.total_steps - self.global_step,
            desc="Video DiT Training",
            initial=0,
        )

        while self.global_step < self.cfg.train.total_steps:
            for batch in loader:
                if self.global_step >= self.cfg.train.total_steps:
                    break

                metrics = self.train_step(batch)
                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % self.cfg.train.log_every == 0:
                    pbar.set_postfix(
                        step=self.global_step,
                        loss=f"{metrics['loss']:.4f}",
                        lr=f"{metrics['lr']:.2e}",
                    )

                # Checkpointing
                if self.global_step % self.cfg.train.save_every == 0:
                    self.save_checkpoint()

        pbar.close()
        self.save_checkpoint()
        logger.info("Training complete. Final step: %d", self.global_step)

    def save_checkpoint(self):
        """Save model, EMA, optimizer, and step."""
        path = self.output_dir / f"video_dit_step_{self.global_step}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": {
                "d_model": self.cfg.model.d_model,
                "depth": self.cfg.model.depth,
                "heads": self.cfg.model.heads,
                "num_frames": self.cfg.video.num_frames,
                "num_cond_frames": self.cfg.video.num_cond_frames,
            },
        }, path)
        logger.info("Saved checkpoint to %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Video DiT for agent sequences")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--ae-ckpt", type=str, default=None,
                        help="Path to pretrained AE checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Video DiT checkpoint to resume from")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf overrides (e.g. train.lr=1e-4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    cfg = load_config(args.config, args.overrides)

    trainer = VideoDiTTrainer(cfg)
    if args.ae_ckpt:
        trainer.load_ae_checkpoint(args.ae_ckpt)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
