"""Stage 2: Diffusion model training with constraint losses."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.agent_dit import build_agent_dit
from ..models.autoencoder import AgentAutoencoder
from ..diffusion.scheduler import NoiseScheduler
from ..diffusion.ddpm import DDPMTrainer
from ..constraints.soft_loss import ConstraintLoss
from ..data.dataset import AgentTransitionDataset, SyntheticAgentDataset
from ..utils.config import load_config


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


class DiffusionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Autoencoder (frozen, loaded from Stage 1)
        self.ae = AgentAutoencoder(
            raw_dim=cfg.agent.raw_dim,
            latent_dim=cfg.agent.latent_dim,
        ).to(self.device)
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        # Diffusion model
        self.model = build_agent_dit(cfg).to(self.device)

        # Noise scheduler
        self.scheduler = NoiseScheduler(
            timesteps=cfg.diffusion.timesteps,
            schedule=cfg.diffusion.beta_schedule,
        ).to(self.device)

        # DDPM trainer
        self.ddpm = DDPMTrainer(
            self.model, self.scheduler, cfg.diffusion.prediction_type
        )

        # Constraint loss
        self.constraint_loss = ConstraintLoss(
            lambda_clearing=cfg.constraint.lambda_clearing,
            lambda_budget=cfg.constraint.lambda_budget,
            lambda_conservation=cfg.constraint.lambda_conservation,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.train.total_steps
        )

        # EMA
        self.ema = EMA(self.model, cfg.train.ema_decay)

        self.global_step = 0
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_ae_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.ae.load_state_dict(ckpt["model"])
        print(f"Loaded AE from {path}")

    def build_dataloader(self) -> DataLoader:
        try:
            dataset = AgentTransitionDataset(self.cfg.data.data_dir)
        except FileNotFoundError:
            print("No data files found, using synthetic data for testing")
            dataset = SyntheticAgentDataset(
                num_samples=500,
                grid_h=self.cfg.patch.grid_h,
                grid_w=self.cfg.patch.grid_w,
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
        pbar = tqdm(total=self.cfg.train.total_steps, desc="Diffusion Training")

        while self.global_step < self.cfg.train.total_steps:
            for batch in loader:
                if self.global_step >= self.cfg.train.total_steps:
                    break

                # Move to device
                state_t = batch["state_t"].to(self.device)
                state_t1 = batch["state_t1"].to(self.device)
                market_cond = batch["market_cond"].to(self.device)

                # Encode to latent space
                with torch.no_grad():
                    z0 = self.ae.encode(state_t1)  # target: next state latent

                # Diffusion loss
                ddpm_out = self.ddpm.compute_loss(z0, market_cond)
                diff_loss = ddpm_out["loss"]

                # Constraint loss (on decoded prediction)
                # Approximate: decode the predicted x0 from v-prediction
                with torch.no_grad():
                    if self.cfg.diffusion.prediction_type == "v_prediction":
                        z0_pred = self.scheduler.predict_x0_from_v(
                            ddpm_out["z_t"], ddpm_out["t"], ddpm_out["pred"]
                        )
                    else:
                        z0_pred = ddpm_out["pred"]
                    state_pred = self.ae.decode(z0_pred)

                constraint_out = self.constraint_loss(state_t, state_pred)

                # Total loss
                loss = diff_loss + constraint_out["constraint_total"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.ema.update(self.model)

                self.global_step += 1
                pbar.update(1)

                if self.global_step % self.cfg.train.log_every == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        diff=f"{diff_loss.item():.4f}",
                        clear=f"{constraint_out['clearing_loss'].item():.4f}",
                    )

                if self.global_step % self.cfg.train.save_every == 0:
                    self.save_checkpoint()

        pbar.close()
        self.save_checkpoint()

    def save_checkpoint(self):
        path = self.output_dir / f"dit_step_{self.global_step}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
        }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ae-ckpt", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    trainer = DiffusionTrainer(cfg)
    if args.ae_ckpt:
        trainer.load_ae_checkpoint(args.ae_ckpt)
    trainer.train()


if __name__ == "__main__":
    main()
