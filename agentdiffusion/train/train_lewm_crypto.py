"""LeWorldModel training on Binance kline data.

Adapts the existing LeWM training loop for crypto time-series data:
- Uses CryptoKlineDataset / CryptoSequenceDataset instead of ABIDES agent grids
- Reshapes 1-D sliding windows into [8, 8, feature_dim] grids for patchify
- Replaces agent-grid-specific price extraction with direct feature-based returns
- Adds crypto-specific evaluation (stylized facts on generated rollouts)

The model architecture (encoder/predictor/decoder/SIGReg) stays identical;
only the data pipeline and loss computation for returns changes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.lewm import LeWorldModel, LeWMLossOutput, SIGReg
from ..models.lewm_encoder import LeWMEncoder
from ..models.lewm_predictor import LeWMPredictor
from ..models.lewm_decoder import LeWMDecoder
from ..data.crypto_dataset import CryptoKlineDataset, CryptoSequenceDataset
from ..utils.config import load_config


def build_lewm_crypto(cfg) -> LeWorldModel:
    """Build LeWorldModel with crypto-adapted dimensions.

    Key differences from build_lewm():
    - d_agent = feature_dim (32 instead of 128)
    - grid_h = grid_w = 8 (instead of 36)
    - patch_size = 2 or 4 (configurable)
    """
    lc = cfg.lewm
    p = cfg.patch.patch_size
    pad_h = ((cfg.patch.grid_h + p - 1) // p) * p
    pad_w = ((cfg.patch.grid_w + p - 1) // p) * p

    return LeWorldModel(
        d_agent=cfg.agent.raw_dim,
        d_enc=lc.d_enc,
        d_latent=lc.d_latent,
        d_pred=lc.d_pred,
        d_cond=lc.get("d_cond", 32),
        patch_size=p,
        enc_depth=lc.enc_depth,
        enc_heads=lc.enc_heads,
        pred_depth=lc.pred_depth,
        pred_heads=lc.pred_heads,
        enc_mlp_ratio=lc.get("enc_mlp_ratio", 4.0),
        pred_mlp_ratio=lc.get("pred_mlp_ratio", 2.0),
        num_projections=lc.get("num_projections", 512),
        lambda_sigreg=lc.get("lambda_sigreg", 0.5),
        lambda_recon=lc.get("lambda_recon", 1.0),
        lambda_price=lc.get("lambda_price", 1.0),
        lambda_returns=lc.get("lambda_returns", 1.0),
        beta_leverage=lc.get("beta_leverage", 0.5),
        dropout=lc.get("dropout", 0.0),
        use_decoder=lc.get("use_decoder", True),
        d_dec=lc.get("d_dec", 256),
        dec_depth=lc.get("dec_depth", 4),
        dec_heads=lc.get("dec_heads", 4),
        dec_mlp_ratio=lc.get("dec_mlp_ratio", 4.0),
        dec_grid_h=pad_h,
        dec_grid_w=pad_w,
    )


class CryptoLeWMTrainer:
    """Trainer for LeWorldModel on crypto kline data.

    Differences from LeWMTrainer:
    1. Uses CryptoKlineDataset / CryptoSequenceDataset
    2. Returns are extracted from feature column 0 (log_return) instead of
       agent grid dim 98
    3. Adds periodic stylized-facts evaluation during training
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: LeWorldModel = build_lewm_crypto(cfg).to(self.device)
        self._log_param_count()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.95),
        )

        self.warmup_steps = cfg.train.warmup_steps
        self.total_steps = cfg.train.total_steps
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_steps,
        )

        self.global_step = 0
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Scaler for mixed precision
        self.use_amp = cfg.train.get("mixed_precision", "no") in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if cfg.train.get("mixed_precision", "no") == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # Keep dataset reference for evaluation
        self.dataset: CryptoKlineDataset | None = None

    def _log_param_count(self):
        enc_params = sum(p.numel() for p in self.model.encoder.parameters())
        pred_params = sum(p.numel() for p in self.model.predictor.parameters())
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LeWorldModel (crypto) parameters:")
        print(f"  Encoder:   {enc_params / 1e6:.2f}M")
        print(f"  Predictor: {pred_params / 1e6:.2f}M")
        if self.model.decoder is not None:
            dec_params = sum(p.numel() for p in self.model.decoder.parameters())
            print(f"  Decoder:   {dec_params / 1e6:.2f}M")
        print(f"  Total:     {total / 1e6:.2f}M")

    def _warmup_lr(self):
        if self.global_step < self.warmup_steps:
            warmup_factor = self.global_step / max(1, self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.train.lr * warmup_factor

    def build_dataloader(self) -> DataLoader:
        p = self.cfg.patch.patch_size
        grid_h = self.cfg.patch.grid_h
        grid_w = self.cfg.patch.grid_w
        window_size = grid_h * grid_w
        feature_dim = self.cfg.agent.raw_dim
        cond_dim = self.cfg.lewm.get("d_cond", 32)

        data_dir = self.cfg.data.data_dir
        symbol = self.cfg.data.get("symbol", "BTCUSDT")
        stride = self.cfg.data.get("stride", 1)

        rollout_steps = int(getattr(self.cfg.train, "rollout_steps", 1))
        seq_len = int(getattr(self.cfg.train, "seq_len", 8))

        if rollout_steps > 1:
            dataset = CryptoSequenceDataset(
                data_dir=data_dir,
                symbol=symbol,
                window_size=window_size,
                feature_dim=feature_dim,
                seq_len=seq_len,
                stride=stride,
                cond_dim=cond_dim,
                grid_h=grid_h,
                grid_w=grid_w,
            )
        else:
            dataset = CryptoKlineDataset(
                data_dir=data_dir,
                symbol=symbol,
                window_size=window_size,
                feature_dim=feature_dim,
                stride=stride,
                cond_dim=cond_dim,
                grid_h=grid_h,
                grid_w=grid_w,
            )

        self._rollout_steps = rollout_steps
        # Keep reference for eval
        if isinstance(dataset, CryptoKlineDataset):
            self.dataset = dataset
        else:
            # Build a single-step dataset too for evaluation
            self.dataset = CryptoKlineDataset(
                data_dir=data_dir, symbol=symbol,
                window_size=window_size, feature_dim=feature_dim,
                stride=stride, cond_dim=cond_dim,
                grid_h=grid_h, grid_w=grid_w,
            )

        return DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True,
        )

    def _compute_crypto_loss(
        self,
        state_t: torch.Tensor,
        state_t1: torch.Tensor,
        market_cond: torch.Tensor,
    ) -> LeWMLossOutput:
        """Compute loss adapted for crypto features.

        Instead of extracting price from agent dim 98, we use feature column 0
        (log_return) directly from the state tensor. The state is
        [B, H, W, feature_dim] where the underlying data is a time series
        reshaped into a grid.
        """
        # Encode both states
        z_t = self.model.encoder(state_t)      # [B, d_latent]
        z_t1 = self.model.encoder(state_t1)    # [B, d_latent]

        # Predict next latent
        z_t1_pred = self.model.predict(z_t, market_cond, stochastic=True)

        # Prediction loss
        loss_pred = F.mse_loss(z_t1_pred, z_t1)

        # Anti-collapse regularization
        z_all = torch.cat([z_t, z_t1], dim=0)
        z_std = z_all.std(dim=0)
        loss_var = F.relu(1.0 - z_std).mean()
        z_centered = z_all - z_all.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(z_all.shape[0] - 1, 1)
        off_diag = cov - torch.diag(cov.diag())
        loss_cov = off_diag.pow(2).mean()
        loss_sigreg = self.model.sigreg(z_all)
        loss_reg = loss_var * 2.0 + loss_cov * 0.1 + loss_sigreg * self.model.lambda_sigreg

        loss_total = loss_pred + loss_reg

        # Reconstruction loss
        loss_recon = None
        if self.model.decoder is not None:
            state_t_recon = self.model.decoder(z_t)
            state_t1_recon = self.model.decoder(z_t1)
            loss_recon = 0.5 * (
                F.mse_loss(state_t_recon, state_t)
                + F.mse_loss(state_t1_recon, state_t1)
            )
            loss_total = loss_total + self.model.lambda_recon * loss_recon

        # --- Crypto return distribution loss ---
        # Extract log-return directly from features. In our grid reshape:
        # state [B, H, W, feature_dim] -> flatten back to [B, H*W, feature_dim]
        # Feature column 0 = normalised log_return per bar.
        # We use the mean return across the window as the "price change" signal.
        B = state_t.shape[0]
        ret_t = state_t.reshape(B, -1, state_t.shape[-1])[:, :, 0].mean(dim=1)   # [B]
        ret_t1 = state_t1.reshape(B, -1, state_t1.shape[-1])[:, :, 0].mean(dim=1)  # [B]

        # Ground truth: difference in mean return between windows
        # This captures the directional shift
        ret_gt = ret_t1 - ret_t

        # Predict Student-t parameters
        from torch.distributions import StudentT
        mu, sigma, nu = self.model.return_dist_head(z_t1_pred)

        # Leverage effect
        ret_prev = market_cond[:, 7]  # previous return from cond slot 7
        neg_shock = F.relu(-ret_prev)
        sigma = sigma * (1.0 + self.model.beta_leverage * neg_shock)

        # Student-t NLL
        dist = StudentT(df=nu, loc=mu, scale=sigma)
        nll_raw = -dist.log_prob(ret_gt)
        loss_ret_nll = nll_raw.clamp(min=-10.0, max=10.0).mean()

        # Sigma floor penalty
        loss_sigma_floor = F.relu(0.005 - sigma).mean() * 100.0

        # Whiteness penalty
        residuals = ((ret_gt - mu) / sigma).detach()
        loss_white = residuals.pow(2).mean() * 0.1

        loss_total = (loss_total
                      + self.model.lambda_price * loss_ret_nll
                      + self.model.lambda_returns * loss_white
                      + loss_sigma_floor)

        return LeWMLossOutput(
            loss_total=loss_total,
            loss_pred=loss_pred,
            loss_sigreg=loss_reg,
            loss_recon=loss_recon,
            loss_price=loss_ret_nll,
            loss_returns=loss_white,
            loss_ret_nll=loss_ret_nll,
            z_t=z_t,
            z_t1_target=z_t1,
            z_t1_pred=z_t1_pred,
        )

    def _train_step_single(self, batch: dict[str, torch.Tensor]) -> LeWMLossOutput:
        state_t = batch["state_t"].to(self.device)
        state_t1 = batch["state_t1"].to(self.device)
        market_cond = batch["market_cond"].to(self.device)
        return self._compute_crypto_loss(state_t, state_t1, market_cond)

    def _train_step_rollout(self, batch: dict[str, torch.Tensor]) -> LeWMLossOutput:
        """Multi-step rollout training for crypto sequences."""
        from torch.distributions import StudentT

        seq_states = batch["seq_states"].to(self.device)  # [B, K+1, H, W, C]
        seq_conds = batch["seq_conds"].to(self.device)    # [B, K, cond_dim]

        B, Kp1, H, W, C = seq_states.shape
        K = Kp1 - 1
        rollout_k = min(self._rollout_steps, K)
        gamma = 0.95

        z = self.model.encode(seq_states[:, 0])

        loss_total = torch.tensor(0.0, device=self.device)
        loss_pred_acc = torch.tensor(0.0, device=self.device)
        loss_ret_nll_acc = torch.tensor(0.0, device=self.device)
        loss_white_acc = torch.tensor(0.0, device=self.device)
        loss_recon_acc = torch.tensor(0.0, device=self.device)

        z_t1_pred_last = None
        z_t1_target_last = None

        for k in range(rollout_k):
            cond_k = seq_conds[:, k]
            state_kp1 = seq_states[:, k + 1]
            state_k = seq_states[:, k]

            z_pred = self.model.predict(z, cond_k, stochastic=True)
            z_target = self.model.encode(state_kp1)

            discount = gamma ** k

            # Prediction loss
            step_pred = F.mse_loss(z_pred, z_target)
            loss_pred_acc = loss_pred_acc + discount * step_pred

            # Reconstruction loss
            if self.model.decoder is not None:
                decoded = self.model.decode(z_pred)
                step_recon = F.mse_loss(decoded, state_kp1)
                loss_recon_acc = loss_recon_acc + discount * step_recon

            # Return distribution loss (crypto-adapted)
            ret_k = state_k.reshape(B, -1, C)[:, :, 0].mean(dim=1)
            ret_kp1 = state_kp1.reshape(B, -1, C)[:, :, 0].mean(dim=1)
            ret_gt = ret_kp1 - ret_k

            mu, sigma, nu = self.model.return_dist_head(z_pred)
            ret_prev = cond_k[:, 7]
            neg_shock = F.relu(-ret_prev)
            sigma = sigma * (1.0 + self.model.beta_leverage * neg_shock)

            dist = StudentT(df=nu, loc=mu, scale=sigma)
            step_ret_nll = -dist.log_prob(ret_gt).clamp(min=-10.0, max=10.0).mean()
            loss_ret_nll_acc = loss_ret_nll_acc + discount * step_ret_nll

            residuals = ((ret_gt - mu) / sigma).detach()
            step_white = residuals.pow(2).mean() * 0.1
            loss_white_acc = loss_white_acc + discount * step_white

            z_t1_pred_last = z_pred
            z_t1_target_last = z_target
            z = z_pred

        # Combine
        w_pred = 1.0
        w_recon = self.model.lambda_recon if self.model.decoder is not None else 0.0
        w_ret_nll = self.model.lambda_price
        w_white = self.model.lambda_returns

        loss_total = (w_pred * loss_pred_acc
                      + w_recon * loss_recon_acc
                      + w_ret_nll * loss_ret_nll_acc
                      + w_white * loss_white_acc)

        # SIGReg on final latents
        z_all = torch.cat([self.model.encode(seq_states[:, 0]), z], dim=0)
        z_std_val = z_all.std(dim=0)
        loss_var = F.relu(1.0 - z_std_val).mean()
        z_centered = z_all - z_all.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(z_all.shape[0] - 1, 1)
        off_diag = cov - torch.diag(cov.diag())
        loss_cov = off_diag.pow(2).mean()
        loss_sigreg_val = self.model.sigreg(z_all)
        loss_reg = loss_var * 2.0 + loss_cov * 0.1 + loss_sigreg_val * self.model.lambda_sigreg
        loss_total = loss_total + loss_reg

        return LeWMLossOutput(
            loss_total=loss_total,
            loss_pred=loss_pred_acc,
            loss_sigreg=loss_reg,
            loss_recon=loss_recon_acc if self.model.decoder is not None else None,
            loss_price=loss_ret_nll_acc,
            loss_returns=loss_white_acc,
            loss_ret_nll=loss_ret_nll_acc,
            z_t=self.model.encode(seq_states[:, 0]),
            z_t1_target=z_t1_target_last if z_t1_target_last is not None else torch.zeros(1, device=self.device),
            z_t1_pred=z_t1_pred_last if z_t1_pred_last is not None else torch.zeros(1, device=self.device),
        )

    def train(self):
        loader = self.build_dataloader()
        self.model.train()
        pbar = tqdm(total=self.total_steps, desc="LeWM-Crypto Training")

        use_rollout = self._rollout_steps > 1

        while self.global_step < self.total_steps:
            for batch in loader:
                if self.global_step >= self.total_steps:
                    break

                self._warmup_lr()

                # Forward pass (optionally with AMP)
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        if use_rollout:
                            loss_out = self._train_step_rollout(batch)
                        else:
                            loss_out = self._train_step_single(batch)
                else:
                    if use_rollout:
                        loss_out = self._train_step_rollout(batch)
                    else:
                        loss_out = self._train_step_single(batch)

                # Backward
                self.optimizer.zero_grad()
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.scale(loss_out.loss_total).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_out.loss_total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                    self.optimizer.step()

                if self.global_step >= self.warmup_steps:
                    self.lr_scheduler.step()

                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % self.cfg.train.log_every == 0:
                    with torch.no_grad():
                        z_std = loss_out.z_t.std(dim=0).mean().item()
                        cosine_sim = F.cosine_similarity(
                            loss_out.z_t1_pred, loss_out.z_t1_target, dim=-1,
                        ).mean().item()

                    postfix = dict(
                        loss=f"{loss_out.loss_total.item():.4f}",
                        pred=f"{loss_out.loss_pred.item():.4f}",
                        reg=f"{loss_out.loss_sigreg.item():.4f}",
                        z_std=f"{z_std:.3f}",
                        cos=f"{cosine_sim:.3f}",
                    )
                    if loss_out.loss_recon is not None:
                        postfix["recon"] = f"{loss_out.loss_recon.item():.4f}"
                    if loss_out.loss_ret_nll is not None:
                        postfix["nll"] = f"{loss_out.loss_ret_nll.item():.4f}"
                    pbar.set_postfix(**postfix)

                # Checkpoint
                if self.global_step % self.cfg.train.save_every == 0:
                    self.save_checkpoint()

                # Periodic evaluation
                eval_every = self.cfg.train.get("eval_every", 5000)
                if self.global_step % eval_every == 0:
                    self.evaluate()
                    self.model.train()

        pbar.close()
        self.save_checkpoint()
        self.evaluate()
        print(f"LeWM-Crypto training complete at step {self.global_step}.")

    def evaluate(self, rollout_steps: int = 1000):
        """Run evaluation: generate rollout and check stylized facts."""
        self.model.eval()
        if self.dataset is None:
            print("[eval] No dataset available for evaluation.")
            return

        # Pick a random starting sample
        start_idx = len(self.dataset) // 2
        sample = self.dataset[start_idx]
        state_t = sample["state_t"].unsqueeze(0).to(self.device)
        market_cond = sample["market_cond"].unsqueeze(0).to(self.device)

        # Generate latent rollout
        with torch.no_grad():
            z = self.model.encode(state_t)
            generated_returns = []

            for step in range(rollout_steps):
                z_next = self.model.predict(z, market_cond, stochastic=False)

                # Predict return distribution and sample
                mu, sigma, nu = self.model.return_dist_head(z_next)
                # Use the mean as the generated return
                generated_returns.append(mu.item())

                # If decoder exists, decode and re-extract market_cond
                if self.model.decoder is not None:
                    decoded = self.model.decode(z_next)  # [1, H, W, C]
                    # Update market_cond from decoded features
                    flat = decoded.reshape(1, -1, decoded.shape[-1])
                    # Update return signals in market_cond
                    mean_ret = flat[0, :, 0].mean().item()
                    market_cond_np = market_cond.cpu().numpy()[0]
                    market_cond_np[0] = mean_ret  # 1-bar return
                    market_cond_np[7] = mean_ret  # prev return
                    market_cond = torch.from_numpy(market_cond_np).unsqueeze(0).float().to(self.device)

                z = z_next

        # Convert generated returns to price series
        gen_returns = np.array(generated_returns)
        gen_prices = np.exp(np.cumsum(gen_returns * 0.01)) * 40000.0  # scale to ~BTC range

        # Get ground truth for comparison
        gt_prices = self.dataset.get_close_prices(start_idx, rollout_steps + 64)
        gt_volumes = self.dataset.get_volumes(start_idx, rollout_steps + 64)

        # Evaluate stylized facts
        try:
            from ..eval.stylized_facts import evaluate_stylized_facts
            gen_report = evaluate_stylized_facts(gen_prices)
            gt_report = evaluate_stylized_facts(gt_prices[:len(gen_prices)], gt_volumes[:len(gen_prices)])

            print(f"\n[eval step {self.global_step}] Stylized Facts:")
            print(f"  Generated: {gen_report.summary}")
            print(f"    fat_tail={gen_report.fat_tail_pass} (alpha={gen_report.fat_tail_alpha:.2f})")
            print(f"    vol_cluster={gen_report.volatility_clustering_pass}")
            print(f"    leverage={gen_report.leverage_effect_pass} (corr={gen_report.leverage_effect_corr:.4f})")
            print(f"    ret_autocorr={gen_report.return_autocorr_pass}")
            print(f"    gain_loss={gen_report.gain_loss_asymmetry_pass}")
            print(f"  Ground truth: {gt_report.summary}")
        except Exception as e:
            print(f"[eval] stylized facts eval failed: {e}")

        # Latent space stats
        with torch.no_grad():
            sample2 = self.dataset[start_idx + 100]
            z2 = self.model.encode(sample2["state_t"].unsqueeze(0).to(self.device))
            z_norms = z.norm(dim=-1).mean().item()
            z2_norms = z2.norm(dim=-1).mean().item()
            print(f"  Latent norm (end of rollout): {z_norms:.2f}")
            print(f"  Latent norm (fresh encode):   {z2_norms:.2f}")
            print(f"  Generated return std: {gen_returns.std():.6f}")
            print(f"  Generated return mean: {gen_returns.mean():.6f}")

    def save_checkpoint(self):
        path = self.output_dir / f"lewm_crypto_step_{self.global_step}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "step": self.global_step,
            "config": {
                "d_latent": self.model.d_latent,
                "grid_h": self.cfg.patch.grid_h,
                "grid_w": self.cfg.patch.grid_w,
                "feature_dim": self.cfg.agent.raw_dim,
                "patch_size": self.cfg.patch.patch_size,
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
    parser = argparse.ArgumentParser(description="Train LeWorldModel on Binance crypto data")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf-style overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    trainer = CryptoLeWMTrainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
