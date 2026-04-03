"""Stage 1: Autoencoder pretraining."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.autoencoder import AgentAutoencoder
from ..data.dataset import AgentFlatDataset, SyntheticAgentDataset
from ..utils.config import load_config


class AETrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = AgentAutoencoder(
            raw_dim=cfg.agent.raw_dim,
            latent_dim=cfg.agent.latent_dim,
            vae=False,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.train.total_steps
        )

        self.global_step = 0
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_dataloader(self) -> DataLoader:
        try:
            dataset = AgentFlatDataset(
                self.cfg.data.data_dir,
                self.cfg.patch.grid_h,
                self.cfg.patch.grid_w,
            )
        except FileNotFoundError:
            print("No data files found, using synthetic data for testing")
            synthetic = SyntheticAgentDataset(
                num_samples=500,
                grid_h=self.cfg.patch.grid_h,
                grid_w=self.cfg.patch.grid_w,
                raw_dim=self.cfg.agent.raw_dim,
            )

            class _FlatWrapper:
                def __init__(self, ds):
                    self.ds = ds
                    H, W, C = ds.grid_h, ds.grid_w, ds.raw_dim
                    self.agents_per = H * W
                def __len__(self):
                    return len(self.ds) * self.agents_per
                def __getitem__(self, idx):
                    file_idx = idx // self.agents_per
                    agent_idx = idx % self.agents_per
                    data = self.ds[file_idx]
                    flat = data["state_t"].reshape(-1, data["state_t"].shape[-1])
                    return flat[agent_idx]

            dataset = _FlatWrapper(synthetic)

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
        pbar = tqdm(total=self.cfg.train.total_steps, desc="AE Training")

        while self.global_step < self.cfg.train.total_steps:
            for batch in loader:
                if self.global_step >= self.cfg.train.total_steps:
                    break

                batch = batch.to(self.device)
                out = self.model(batch)
                loss = out["recon_loss"]
                if "kl_loss" in out:
                    loss = loss + 0.01 * out["kl_loss"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                self.global_step += 1
                pbar.update(1)

                if self.global_step % self.cfg.train.log_every == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.6f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    )

                if self.global_step % self.cfg.train.save_every == 0:
                    self.save_checkpoint()

        pbar.close()
        self.save_checkpoint()
        print(f"AE training complete. Final loss: {loss.item():.6f}")

    def save_checkpoint(self):
        path = self.output_dir / f"ae_step_{self.global_step}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
        }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    trainer = AETrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
