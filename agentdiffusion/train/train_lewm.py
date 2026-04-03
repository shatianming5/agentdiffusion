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

from ..models.lewm import build_lewm, LeWorldModel, masked_mean
from ..data.dataset import AgentTransitionDataset, AgentSequenceDataset, SyntheticAgentDataset
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

        rollout_steps = int(getattr(self.cfg.train, "rollout_steps", 1))
        seq_len = int(getattr(self.cfg.train, "seq_len", 8))

        if rollout_steps > 1:
            # Use sequence dataset for multi-step rollout training
            try:
                dataset = AgentSequenceDataset(
                    self.cfg.data.data_dir,
                    seq_len=seq_len,
                    pad_to=(pad_h, pad_w),
                )
                print(f"Using AgentSequenceDataset: seq_len={seq_len}, "
                      f"rollout_steps={rollout_steps}, {len(dataset)} windows")
            except (FileNotFoundError, ValueError) as e:
                print(f"Sequence dataset failed ({e}), falling back to single-step")
                rollout_steps = 1
                dataset = self._build_single_step_dataset(pad_h, pad_w)
        else:
            dataset = self._build_single_step_dataset(pad_h, pad_w)

        self._rollout_steps = rollout_steps
        return DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True,
        )

    def _build_single_step_dataset(self, pad_h: int, pad_w: int):
        """Build the standard single-step AgentTransitionDataset."""
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
        return dataset

    def _train_step_single(self, batch: dict[str, torch.Tensor]):
        """Standard single-step training: one (state_t, state_t1) pair."""
        state_t = batch["state_t"].to(self.device)
        state_t1 = batch["state_t1"].to(self.device)
        market_cond = batch["market_cond"].to(self.device)
        return self.model.compute_loss(state_t, state_t1, market_cond)

    def _train_step_rollout(self, batch: dict[str, torch.Tensor]):
        """Multi-step unrolled training over K consecutive transitions.

        Accumulates prediction, reconstruction, and price losses
        at each step with temporal discounting (gamma^k).
        """
        import torch.nn.functional as F_fn

        seq_states = batch["seq_states"].to(self.device)   # [B, K+1, H, W, C]
        seq_conds = batch["seq_conds"].to(self.device)     # [B, K, cond_dim]

        B, Kp1, H, W, C = seq_states.shape
        K = Kp1 - 1
        rollout_k = min(self._rollout_steps, K)
        gamma = 0.95

        # Encode the first state
        z = self.model.encode(seq_states[:, 0])  # [B, d_latent]

        loss_total = torch.tensor(0.0, device=self.device)
        loss_pred_acc = torch.tensor(0.0, device=self.device)
        loss_price_acc = torch.tensor(0.0, device=self.device)
        loss_recon_acc = torch.tensor(0.0, device=self.device)

        z_t1_pred_last = None
        z_t1_target_last = None

        for k in range(rollout_k):
            cond_k = seq_conds[:, k]                    # [B, cond_dim]
            state_kp1 = seq_states[:, k + 1]            # [B, H, W, C]
            state_k = seq_states[:, k]                  # [B, H, W, C]

            z_pred = self.model.predictor(z, cond_k)    # [B, d_latent]
            z_target = self.model.encode(state_kp1)     # [B, d_latent]

            discount = gamma ** k

            # Prediction loss in latent space
            step_pred = F_fn.mse_loss(z_pred, z_target)
            loss_pred_acc = loss_pred_acc + discount * step_pred

            # Reconstruction loss (if decoder available)
            if self.model.decoder is not None:
                decoded = self.model.decode(z_pred)
                step_recon = F_fn.mse_loss(decoded, state_kp1)
                loss_recon_acc = loss_recon_acc + discount * step_recon

            # Price prediction loss with masked mean
            mask_hw = (state_k[..., 0] != 0) | (state_k[..., 1] != 0)  # [B, H, W]
            price_kp1_gt = masked_mean(state_kp1[..., 98], mask_hw, dims=(1, 2))  # [B]
            price_kp1_pred = self.model.price_head(z_pred).squeeze(-1)             # [B]
            step_price = F_fn.mse_loss(price_kp1_pred, price_kp1_gt)
            loss_price_acc = loss_price_acc + discount * step_price

            z_t1_pred_last = z_pred
            z_t1_target_last = z_target

            # Use predicted latent for next step (autoregressive rollout)
            z = z_pred

        # Combine accumulated losses
        w_pred = 1.0
        w_recon = self.model.lambda_recon if self.model.decoder is not None else 0.0
        w_price = self.model.lambda_price

        loss_total = (w_pred * loss_pred_acc
                      + w_recon * loss_recon_acc
                      + w_price * loss_price_acc)

        # SIGReg on final latents (do once, not per step)
        z_all = torch.cat([self.model.encode(seq_states[:, 0]), z], dim=0)
        z_std = z_all.std(dim=0)
        loss_var = F_fn.relu(1.0 - z_std).mean()
        z_centered = z_all - z_all.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(z_all.shape[0] - 1, 1)
        off_diag = cov - torch.diag(cov.diag())
        loss_cov = off_diag.pow(2).mean()
        loss_sigreg_val = self.model.sigreg(z_all)
        loss_reg = loss_var * 2.0 + loss_cov * 0.1 + loss_sigreg_val * self.model.lambda_sigreg
        loss_total = loss_total + loss_reg

        # Return a compatible LeWMLossOutput-like object
        from ..models.lewm import LeWMLossOutput
        return LeWMLossOutput(
            loss_total=loss_total,
            loss_pred=loss_pred_acc,
            loss_sigreg=loss_reg,
            loss_recon=loss_recon_acc if self.model.decoder is not None else None,
            loss_price=loss_price_acc,
            loss_returns=None,
            z_t=self.model.encode(seq_states[:, 0]),
            z_t1_target=z_t1_target_last if z_t1_target_last is not None else torch.zeros(1, device=self.device),
            z_t1_pred=z_t1_pred_last if z_t1_pred_last is not None else torch.zeros(1, device=self.device),
        )

    def train(self):
        loader = self.build_dataloader()
        self.model.train()
        pbar = tqdm(total=self.total_steps, desc="LeWM Training")

        use_rollout = self._rollout_steps > 1

        while self.global_step < self.total_steps:
            for batch in loader:
                if self.global_step >= self.total_steps:
                    break

                # Apply warmup
                self._warmup_lr()

                # Compute loss (single-step or multi-step rollout)
                if use_rollout:
                    loss_out = self._train_step_rollout(batch)
                else:
                    loss_out = self._train_step_single(batch)

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
                    if loss_out.loss_price is not None:
                        postfix["price"] = f"{loss_out.loss_price.item():.4f}"
                    if loss_out.loss_returns is not None:
                        postfix["rets"] = f"{loss_out.loss_returns.item():.4f}"
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
