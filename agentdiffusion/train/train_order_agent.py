"""Stage B: Train Order→Agent Encoder + Agent→Order Decoder.

Self-supervised loop on real LOBSTER data:
  orders → Encoder → agent grid → (frozen Video DiT) → future agent grid → Decoder → predicted orders
  Loss = reconstruction on order flow

Usage:
    python -m agentdiffusion.train.train_order_agent \
        --vdit-ckpt outputs/video_dit_10k_noae_constrained/video_dit_step_20000.pt \
        --ob-path data/external/lobster/AMZN_..._orderbook_10.csv \
        --msg-path data/external/lobster/AMZN_..._message_10.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.video_dit import VideoDiT, VideoDDIMSampler
from ..models.order_encoder import OrderToAgentEncoder
from ..models.order_decoder import AgentToOrderDecoder, OrderFlowLoss
from ..diffusion.scheduler import NoiseScheduler
from ..data.lob_order_dataset import LOBOrderFlowDataset

logger = logging.getLogger(__name__)


class OrderAgentTrainer:
    """Train Encoder + Decoder with frozen Video DiT in the middle."""

    def __init__(
        self,
        vdit_ckpt: str | None,
        ob_path: str,
        msg_path: str,
        # Architecture
        d_order: int = 6,
        d_embed: int = 128,
        d_state: int = 16,
        d_order_out: int = 6,
        grid_h: int = 16,
        grid_w: int = 16,
        n_slot_iters: int = 3,
        # Video DiT params (must match checkpoint)
        d_model: int = 256,
        depth: int = 6,
        heads: int = 4,
        patch_size: int = 4,
        num_frames: int = 20,
        num_cond_frames: int = 4,
        # Data
        window_seconds: float = 1.0,
        n_max_orders: int = 256,
        # Training
        lr: float = 3e-4,
        total_steps: int = 5000,
        batch_size: int = 8,
        log_every: int = 50,
        save_every: int = 1000,
        output_dir: str = "outputs/order_agent",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_steps = total_steps
        self.log_every = log_every
        self.save_every = save_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_cond_frames = num_cond_frames

        # --- Frozen Video DiT (optional) ---
        self.skip_vdit = vdit_ckpt is None
        if not self.skip_vdit:
            self.vdit = VideoDiT(
                d_latent=d_state, d_model=d_model, depth=depth, heads=heads,
                patch_size=patch_size, grid_h=grid_h, grid_w=grid_w,
                num_frames=num_frames, num_cond_frames=num_cond_frames,
            ).to(self.device)

            ckpt = torch.load(vdit_ckpt, map_location=self.device, weights_only=True)
            state = ckpt.get("ema", ckpt.get("model"))
            self.vdit.load_state_dict(state)
            self.vdit.eval()
            for p in self.vdit.parameters():
                p.requires_grad_(False)
            logger.info("Loaded frozen Video DiT from %s", vdit_ckpt)

            self.scheduler = NoiseScheduler(1000, "cosine").to(self.device)
            self.sampler = VideoDDIMSampler(
                self.vdit, self.scheduler, "v_prediction", ddim_steps=20, eta=0.0,
            )
        else:
            logger.info("Skipping Video DiT (encoder-decoder only mode)")

        # --- Trainable Encoder + Decoder ---
        self.encoder = OrderToAgentEncoder(
            d_order=d_order, d_embed=d_embed, d_state=d_state,
            grid_h=grid_h, grid_w=grid_w, n_slot_iters=n_slot_iters,
        ).to(self.device)

        self.decoder = AgentToOrderDecoder(
            d_state=d_state, d_hidden=d_embed, d_order_out=d_order_out,
        ).to(self.device)

        self.order_loss = OrderFlowLoss()

        # --- Dataset ---
        self.dataset = LOBOrderFlowDataset(
            ob_path, msg_path,
            window_seconds=window_seconds,
            total_windows=num_frames,
            n_max_orders=n_max_orders,
            d_order=d_order,
        )
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )

        # --- Optimizer (only encoder + decoder params) ---
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps,
        )

        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        logger.info("Encoder params: %.1fM, Decoder params: %.1fM", n_enc/1e6, n_dec/1e6)
        logger.info("Dataset: %d sequences", len(self.dataset))

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One training step of the Order-Agent-Order loop."""
        orders = batch["orders"].to(self.device)  # [B, T, N_max, d_order]
        B, T, N_max, D_ord = orders.shape
        K = self.num_cond_frames

        # --- Encode each window's orders into agent grid ---
        # Flatten [B, T] for encoder
        orders_flat = orders.reshape(B * T, N_max, D_ord)
        grids_flat = self.encoder(orders_flat)  # [B*T, H, W, d_state]
        H, W, d_state = grids_flat.shape[1], grids_flat.shape[2], grids_flat.shape[3]
        grids = grids_flat.reshape(B, T, H, W, d_state)  # [B, T, H, W, d_state]

        # --- Split into condition and generation ---
        grids_cond = grids[:, :K]   # [B, K, H, W, d_state]

        if not self.skip_vdit:
            # Run frozen Video DiT: generate from condition
            N_gen = T - K
            gen_shape = (B, N_gen, H, W, d_state)
            with torch.no_grad():
                grids_pred = self.sampler.sample(
                    grids_cond, gen_shape, device=self.device,
                    zero_sum_proj=True,
                )
            pred_seq = torch.cat([grids_cond[:, -1:], grids_pred], dim=1)
        else:
            # Skip DiT: add noise to encoded grids (denoising autoencoder mode)
            noise_scale = 0.3
            noisy_grids = grids[:, K-1:] + noise_scale * torch.randn_like(grids[:, K-1:])
            pred_seq = noisy_grids  # [B, N+1, H, W, d_state]

        # --- Decode: grids → order predictions ---
        pred_orders = self.decoder.decode_sequence(pred_seq)

        # --- Ground truth: decoder on CLEAN encoded grids ---
        with torch.no_grad():
            gt_orders = self.decoder.decode_sequence(grids[:, K-1:].detach())

        # --- Losses ---
        # 1) Reconstruction: decoded orders should match ground truth decoded orders
        loss_recon = F.mse_loss(pred_orders, gt_orders)

        # 2) Consistency: encoded grids should be temporally smooth
        grid_delta = grids[:, 1:] - grids[:, :-1]
        loss_smooth = grid_delta.pow(2).mean()

        # 3) Diversity: different slots should specialise (entropy regularisation)
        grid_var = grids.var(dim=(0, 1))  # [H, W, d_state]
        loss_diversity = -grid_var.mean()  # maximise variance across cells

        loss = loss_recon + 0.1 * loss_smooth + 0.01 * loss_diversity

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        return {
            "loss": loss.item(),
            "recon": loss_recon.item(),
            "smooth": loss_smooth.item(),
            "diversity": loss_diversity.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train(self):
        """Full training loop."""
        step = 0
        pbar = tqdm(total=self.total_steps, desc="Order-Agent Training")

        while step < self.total_steps:
            for batch in self.loader:
                if step >= self.total_steps:
                    break

                metrics = self.train_step(batch)
                step += 1
                pbar.update(1)

                if step % self.log_every == 0:
                    pbar.set_postfix(
                        step=step,
                        loss=f"{metrics['loss']:.4f}",
                        recon=f"{metrics['recon']:.4f}",
                        lr=f"{metrics['lr']:.2e}",
                    )

                if step % self.save_every == 0:
                    path = self.output_dir / f"order_agent_step_{step}.pt"
                    torch.save({
                        "encoder": self.encoder.state_dict(),
                        "decoder": self.decoder.state_dict(),
                        "step": step,
                    }, path)
                    logger.info("Saved to %s", path)

        pbar.close()
        # Final save
        path = self.output_dir / f"order_agent_step_{step}.pt"
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "step": step,
        }, path)
        logger.info("Training complete. Saved to %s", path)


def main():
    parser = argparse.ArgumentParser(description="Train Order-Agent Encoder+Decoder")
    parser.add_argument("--vdit-ckpt", default=None, help="Frozen Video DiT checkpoint (omit to skip DiT)")
    parser.add_argument("--ob-path", required=True, help="LOBSTER orderbook CSV")
    parser.add_argument("--msg-path", required=True, help="LOBSTER message CSV")
    parser.add_argument("--grid-h", type=int, default=16)
    parser.add_argument("--grid-w", type=int, default=16)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="outputs/order_agent")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    trainer = OrderAgentTrainer(
        vdit_ckpt=args.vdit_ckpt,
        ob_path=args.ob_path,
        msg_path=args.msg_path,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        d_state=args.d_state,
        d_embed=args.d_embed,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
