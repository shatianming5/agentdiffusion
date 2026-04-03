"""LeWorldModel training: JEPA-based next-state prediction in latent space.

Uses the same AgentTransitionDataset as diffusion training. Encodes S_t and
S_{t+1} through the shared encoder, predicts z_{t+1} from z_t + market_cond,
and optimizes MSE prediction loss + SIGReg regularization.

No autoencoder dependency, no EMA, no stop-gradient -- trained end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.lewm import build_lewm, LeWorldModel
from ..data.dataset import AgentTransitionDataset, SyntheticAgentDataset
from ..utils.config import load_config


class LeWMTrainer:
    """Trainer for LeWorldModel."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model: LeWorldModel = build_lewm(cfg).to(self.device)
        self._log_param_count()

        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine annealing with warmup
        self.warmup_steps = cfg.train.warmup_steps
        self.total_steps = cfg.train.total_steps
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_steps,
        )

        self.global_step = 0
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _log_param_count(self):
        enc_params = sum(p.numel() for p in self.model.encoder.parameters())
        pred_params = sum(p.numel() for p in self.model.predictor.parameters())
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LeWorldModel parameters:")
        print(f"  Encoder:   {enc_params / 1e6:.2f}M")
        print(f"  Predictor: {pred_params / 1e6:.2f}M")
        if self.model.decoder is not None:
            dec_params = sum(p.numel() for p in self.model.decoder.parameters())
            print(f"  Decoder:   {dec_params / 1e6:.2f}M")
        print(f"  Total:     {total / 1e6:.2f}M")

    def _warmup_lr(self):
        """Apply linear warmup to learning rate."""
        if self.global_step < self.warmup_steps:
            warmup_factor = self.global_step / max(1, self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.train.lr * warmup_factor

    def build_dataloader(self) -> DataLoader:
        # Compute pad target: round grid up to nearest multiple of patch_size
        p = self.cfg.patch.patch_size
        pad_h = ((self.cfg.patch.grid_h + p - 1) // p) * p
        pad_w = ((self.cfg.patch.grid_w + p - 1) // p) * p
        print(f"Grid {self.cfg.patch.grid_h}x{self.cfg.patch.grid_w} "
              f"-> padded to {pad_h}x{pad_w} (patch_size={p})")

        try:
            dataset = AgentTransitionDataset(
                self.cfg.data.data_dir, pad_to=(pad_h, pad_w),
            )
        except FileNotFoundError:
            print("No data files found, using synthetic data for testing")
            dataset = SyntheticAgentDataset(
                num_samples=500,
                grid_h=pad_h,
                grid_w=pad_w,
                raw_dim=self.cfg.agent.raw_dim,
            )

        return DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True,
        )

    def train(self):
        loader = self.build_dataloader()
        self.model.train()
        pbar = tqdm(total=self.total_steps, desc="LeWM Training")

        while self.global_step < self.total_steps:
            for batch in loader:
                if self.global_step >= self.total_steps:
                    break

                # Move to device
                state_t = batch["state_t"].to(self.device)
                state_t1 = batch["state_t1"].to(self.device)
                market_cond = batch["market_cond"].to(self.device)

                # Apply warmup
                self._warmup_lr()

                # Compute loss
                loss_out = self.model.compute_loss(state_t, state_t1, market_cond)

                # Backprop
                self.optimizer.zero_grad()
                loss_out.loss_total.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.grad_clip,
                )
                self.optimizer.step()

                # Step LR scheduler (after warmup)
                if self.global_step >= self.warmup_steps:
                    self.lr_scheduler.step()

                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % self.cfg.train.log_every == 0:
                    # Compute latent stats for monitoring
                    with torch.no_grad():
                        z_std = loss_out.z_t.std(dim=0).mean().item()
                        z_pred_mse = loss_out.loss_pred.item()
                        cosine_sim = nn.functional.cosine_similarity(
                            loss_out.z_t1_pred, loss_out.z_t1_target, dim=-1,
                        ).mean().item()

                    postfix = dict(
                        loss=f"{loss_out.loss_total.item():.4f}",
                        pred=f"{z_pred_mse:.4f}",
                        sigreg=f"{loss_out.loss_sigreg.item():.4f}",
                        z_std=f"{z_std:.3f}",
                        cos=f"{cosine_sim:.3f}",
                    )
                    if loss_out.loss_recon is not None:
                        postfix["recon"] = f"{loss_out.loss_recon.item():.4f}"
                    pbar.set_postfix(**postfix)

                # Checkpoint
                if self.global_step % self.cfg.train.save_every == 0:
                    self.save_checkpoint()

        pbar.close()
        self.save_checkpoint()
        print(f"LeWM training complete at step {self.global_step}.")

    def save_checkpoint(self):
        path = self.output_dir / f"lewm_step_{self.global_step}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "step": self.global_step,
            "config": {
                "d_latent": self.model.d_latent,
                "lambda_sigreg": self.model.lambda_sigreg,
            },
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["scheduler"])
        if "step" in ckpt:
            self.global_step = ckpt["step"]
        print(f"Resumed from {path} at step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train LeWorldModel")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf-style overrides (e.g. train.lr=3e-4)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    trainer = LeWMTrainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
