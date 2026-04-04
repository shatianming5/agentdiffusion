"""LeWorldModel: JEPA-based world model for agent-based financial simulation.

Combines a ViT encoder and Transformer predictor with SIGReg regularization
for learning next-state prediction in latent space. Optionally includes a
Transformer decoder for reconstructing raw agent grids from latent vectors.

Reference: arXiv:2603.19312 (Le World Model)

Training objective (with decoder):
    L_total = L_pred + lambda_sigreg * L_reg + lambda_recon * L_recon

    L_pred:   MSE between predicted and target latent embeddings
    L_reg:    VICReg-style variance/covariance + SIGReg regularization
    L_recon:  MSE between decoded encoder output and original state (raw space)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lewm_encoder import LeWMEncoder
from .lewm_predictor import LeWMPredictor
from .lewm_decoder import LeWMDecoder
from ..utils.masked import masked_mean


@dataclass
class LeWMLossOutput:
    """Container for LeWorldModel loss components."""
    loss_total: torch.Tensor
    loss_pred: torch.Tensor
    loss_sigreg: torch.Tensor
    loss_recon: torch.Tensor | None
    loss_price: torch.Tensor | None       # repurposed: now holds return MSE contribution
    loss_returns: torch.Tensor | None     # repurposed: now holds volatility matching contribution
    loss_ret_mse: torch.Tensor | None     # MSE on predicted vs actual returns
    loss_vol: torch.Tensor | None         # MSE between predicted sigma and realised |return|
    z_t: torch.Tensor
    z_t1_target: torch.Tensor
    z_t1_pred: torch.Tensor


class SIGReg(nn.Module):
    """SIGReg: Stochastic Independence via Gaussian Regularization.

    Projects embedding batch onto M random unit directions, then
    computes the Epps-Pulley (EP) test statistic measuring departure
    from Gaussianity on each projection. The loss encourages the
    embedding distribution to be Gaussian along all random directions,
    which is a sufficient condition for multivariate Gaussianity.

    This prevents representation collapse without requiring EMA targets,
    stop-gradient, or contrastive pairs.

    Args:
        d_latent: Embedding dimension.
        num_projections: Number of random projection directions (M=512).
    """

    def __init__(self, d_latent: int = 256, num_projections: int = 512):
        super().__init__()
        self.d_latent = d_latent
        self.num_projections = num_projections

        # Random projection directions (fixed, not trained)
        # Each column is a unit direction in R^d_latent
        directions = torch.randn(d_latent, num_projections)
        directions = F.normalize(directions, dim=0)
        self.register_buffer("directions", directions)

    @staticmethod
    def _epps_pulley_statistic(samples: torch.Tensor) -> torch.Tensor:
        """Compute the Epps-Pulley test statistic for univariate normality.

        The EP statistic measures how much the empirical characteristic
        function deviates from the Gaussian characteristic function.
        For a standard normal, the statistic is 0.

        Args:
            samples: [B, M] projected samples (B = batch, M = directions).

        Returns:
            ep_stat: [M] EP statistic per projection direction.
        """
        B, M = samples.shape

        # Standardize each projection to zero mean, unit variance
        mu = samples.mean(dim=0, keepdim=True)  # [1, M]
        std = samples.std(dim=0, keepdim=True).clamp(min=1e-8)  # [1, M]
        y = (samples - mu) / std  # [B, M]

        # EP test statistic:
        # T = (1/B) * sum_i sum_j exp(-0.5*(y_i - y_j)^2)
        #   - 2 * sum_i exp(-0.5 * y_i^2) * sqrt(2)
        #   + B * sqrt(2)
        # Simplified using pairwise differences via broadcasting.
        #
        # More efficient formulation:
        # T = mean_{i,j} exp(-(y_i-y_j)^2 / 2) - 2*mean_i exp(-y_i^2/2)*sqrt(2/(1+1)) + sqrt(2/(2+1))
        # From Epps & Pulley (1983), adapted for computational efficiency.

        # Term 1: mean of pairwise Gaussian kernel
        # Use the identity: E[exp(-(yi-yj)^2/2)] can be computed via
        # exp(-yi^2/2) convolution, but for moderate B we compute directly.
        # For efficiency, use: sum exp(-(yi-yj)^2/2) = |sum exp(-iw*yj)|^2
        # evaluated at appropriate frequencies. Instead, we use the CF approach:

        # Characteristic function approach (efficient for large B):
        # phi_n(t) = (1/B) sum_j exp(i*t*y_j)
        # EP_stat = integral |phi_n(t) - exp(-t^2/2)|^2 w(t) dt

        # For computational tractability, we use the closed-form EP statistic:
        # T_n = (1/n^2) sum_{j,k} exp(-(y_j-y_k)^2/4)
        #     - (2/n) * sqrt(2/3) * sum_j exp(-y_j^2/6)
        #     + sqrt(1/2)

        # Term 1: pairwise kernel (quadratic in B, so we cap computation)
        if B <= 512:
            # Direct pairwise computation
            diff = y.unsqueeze(0) - y.unsqueeze(1)  # [B, B, M]
            term1 = torch.exp(-diff.pow(2) / 4.0).mean(dim=(0, 1))  # [M]
        else:
            # Subsample for efficiency
            idx = torch.randperm(B, device=y.device)[:512]
            y_sub = y[idx]
            diff = y_sub.unsqueeze(0) - y_sub.unsqueeze(1)
            term1 = torch.exp(-diff.pow(2) / 4.0).mean(dim=(0, 1))

        # Term 2: marginal kernel
        term2 = math.sqrt(2.0 / 3.0) * torch.exp(-y.pow(2) / 6.0).mean(dim=0)  # [M]

        # Term 3: constant
        term3 = math.sqrt(0.5)

        ep_stat = term1 - 2.0 * term2 + term3  # [M]

        return ep_stat

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss on a batch of embeddings.

        Args:
            z: Embeddings [B, d_latent].

        Returns:
            loss: Scalar SIGReg loss (mean EP statistic across projections).
        """
        # Project embeddings onto random directions: [B, d_latent] @ [d_latent, M] -> [B, M]
        projections = z @ self.directions  # [B, M]

        # Compute EP statistic per direction
        ep_stats = self._epps_pulley_statistic(projections)  # [M]

        # Loss is the mean absolute EP statistic
        # (EP = 0 means Gaussian, any deviation is penalized)
        loss = ep_stats.abs().mean()

        return loss


class ReturnDistributionHead(nn.Module):
    """Predicts Student-t return distribution parameters from context.

    Outputs (mu, sigma, nu) for a Student-t distribution:
      - mu: location (expected return)
      - sigma: scale (volatility), via softplus to ensure positivity
      - nu: degrees of freedom (> 2 for finite variance), via softplus + 2
    """

    def __init__(self, d_input: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),  # mu, log_sigma, log_nu
        )

    def forward(self, ctx: torch.Tensor, sigma_floor: float = 0.01):
        """Predict return distribution parameters.

        Args:
            ctx: Context vector [B, d_input].
            sigma_floor: Minimum sigma value (default 0.01, caller may
                         pass data_std-based floor for better calibration).

        Returns:
            mu:    [B] location parameter
            sigma: [B] scale parameter (>= sigma_floor)
            nu:    [B] degrees of freedom (> 2)
        """
        out = self.net(ctx)
        mu, log_sigma, log_nu = out.chunk(3, dim=-1)
        floor_t = torch.tensor(sigma_floor, device=out.device, dtype=out.dtype)
        sigma = torch.maximum(F.softplus(log_sigma), floor_t).clamp(max=5.0)
        nu = 2.0 + F.softplus(log_nu)  # df > 2 for finite variance
        return mu.squeeze(-1), sigma.squeeze(-1), nu.squeeze(-1)


class LeWorldModel(nn.Module):
    """Full LeWorldModel: Encoder + Predictor + SIGReg + optional Decoder.

    End-to-end trainable world model that learns to predict next
    agent grid states in latent space. No EMA, no stop-gradient,
    no pre-trained encoder -- trained from scratch.

    When use_decoder=True, includes a Transformer decoder that maps
    latent vectors back to raw agent grids, enabling:
    - Reconstruction loss (L_recon) for better encoder grounding
    - Latent-to-raw-space rollout and evaluation

    Args:
        d_agent: Per-agent raw feature dimension (128).
        d_enc: Encoder hidden dimension (256).
        d_latent: Shared latent space dimension (256).
        d_pred: Predictor hidden dimension (384).
        d_cond: Market condition dimension (32).
        patch_size: Spatial patch size for encoder (4).
        enc_depth: Encoder transformer depth (6).
        enc_heads: Encoder attention heads (8).
        pred_depth: Predictor transformer depth (6).
        pred_heads: Predictor attention heads (8).
        enc_mlp_ratio: FFN expansion ratio for encoder (4.0).
        pred_mlp_ratio: FFN expansion ratio for predictor (2.0).
        num_projections: SIGReg random projection count (512).
        lambda_sigreg: SIGReg loss weight (0.1).
        lambda_recon: Reconstruction loss weight (1.0).
        lambda_price: Return NLL loss weight (10.0, repurposed from price MSE).
        lambda_returns: Whiteness penalty weight (5.0, repurposed from return Huber).
        beta_leverage: Asymmetric volatility leverage effect coefficient (0.5).
        dropout: Dropout rate (0.0).
        use_decoder: Whether to include the decoder (False).
        d_dec: Decoder hidden dimension (256).
        dec_depth: Decoder transformer depth (4).
        dec_heads: Decoder attention heads (4).
        dec_mlp_ratio: FFN expansion ratio for decoder (4.0).
        dec_grid_h: Padded grid height for decoder (36).
        dec_grid_w: Padded grid width for decoder (36).
    """

    def __init__(
        self,
        d_agent: int = 128,
        d_enc: int = 256,
        d_latent: int = 256,
        d_pred: int = 384,
        d_cond: int = 32,
        patch_size: int = 4,
        enc_depth: int = 6,
        enc_heads: int = 8,
        pred_depth: int = 6,
        pred_heads: int = 8,
        enc_mlp_ratio: float = 4.0,
        pred_mlp_ratio: float = 2.0,
        num_projections: int = 512,
        lambda_sigreg: float = 0.1,
        lambda_recon: float = 1.0,
        lambda_price: float = 10.0,
        lambda_returns: float = 5.0,
        beta_leverage: float = 0.5,
        dropout: float = 0.0,
        use_decoder: bool = False,
        d_dec: int = 256,
        dec_depth: int = 4,
        dec_heads: int = 4,
        dec_mlp_ratio: float = 4.0,
        dec_grid_h: int = 36,
        dec_grid_w: int = 36,
    ):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.lambda_recon = lambda_recon
        self.lambda_price = lambda_price      # repurposed: weight for return NLL
        self.lambda_returns = lambda_returns  # repurposed: weight for whiteness penalty
        self.beta_leverage = beta_leverage
        self.d_latent = d_latent
        self.use_decoder = use_decoder

        # Encoder: agent grid -> latent z (~5M params)
        self.encoder = LeWMEncoder(
            d_agent=d_agent,
            d_enc=d_enc,
            d_latent=d_latent,
            patch_size=patch_size,
            depth=enc_depth,
            heads=enc_heads,
            mlp_ratio=enc_mlp_ratio,
            dropout=dropout,
        )

        # Predictor: z_t + cond -> z_{t+1} (~10M params)
        self.predictor = LeWMPredictor(
            d_latent=d_latent,
            d_pred=d_pred,
            d_cond=d_cond,
            depth=pred_depth,
            heads=pred_heads,
            mlp_ratio=pred_mlp_ratio,
            dropout=dropout,
        )

        # SIGReg regularizer
        self.sigreg = SIGReg(d_latent=d_latent, num_projections=num_projections)

        # Return distribution head: latent -> Student-t params (mu, sigma, nu)
        # Replaces the old deterministic price_head
        self.return_dist_head = ReturnDistributionHead(d_input=d_latent, d_hidden=128)

        # Backward-compat alias: price_head now refers to return_dist_head
        # (kept so that external code referencing model.price_head doesn't break)
        self.price_head = self.return_dist_head

        # Latent stochasticity head: predicts per-dim scale for sampling noise
        self.latent_scale_head = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.Softplus(),
        )
        # Initialize to very small scale so training starts near-deterministic
        nn.init.zeros_(self.latent_scale_head[0].weight)
        nn.init.constant_(self.latent_scale_head[0].bias, -3.0)  # softplus(-3) ~ 0.049

        # Optional decoder: latent z -> agent grid
        self.decoder: LeWMDecoder | None = None
        if use_decoder:
            self.decoder = LeWMDecoder(
                d_latent=d_latent,
                d_agent=d_agent,
                d_dec=d_dec,
                patch_size=patch_size,
                grid_h=dec_grid_h,
                grid_w=dec_grid_w,
                depth=dec_depth,
                heads=dec_heads,
                mlp_ratio=dec_mlp_ratio,
                dropout=dropout,
            )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode agent grid to latent vector.

        Args:
            state: Agent grid [B, H, W, d_agent].

        Returns:
            z: Latent vector [B, d_latent].
        """
        return self.encoder(state)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to agent grid.

        Args:
            z: Latent vector [B, d_latent].

        Returns:
            state: Reconstructed agent grid [B, H, W, d_agent].

        Raises:
            RuntimeError: If model was built without decoder (use_decoder=False).
        """
        if self.decoder is None:
            raise RuntimeError(
                "LeWorldModel was built without decoder. "
                "Set use_decoder=True to enable decoding."
            )
        return self.decoder(z)

    def predict(
        self,
        z_t: torch.Tensor,
        market_cond: torch.Tensor,
        causal: bool = False,
        stochastic: bool = False,
    ) -> torch.Tensor:
        """Predict next-step latent from current latent and conditions.

        When stochastic=True (training), adds learned Gaussian noise to the
        predicted mean. When stochastic=False (inference), returns the mean.

        Args:
            z_t: Current latent [B, d_latent] or [B, L, d_latent].
            market_cond: Market conditions [B, d_cond].
            causal: Whether to use causal masking for sequences.
            stochastic: Whether to add latent noise (True during training).

        Returns:
            z_pred: Predicted next latent, same shape as z_t.
        """
        z_mean = self.predictor(z_t, market_cond, causal=causal)

        if stochastic and self.training:
            # Predict per-dim scale from current latent for stochastic sampling
            z_for_scale = z_t if z_t.ndim == 2 else z_t[:, -1]  # use last step if sequence
            z_log_scale = self.latent_scale_head(z_for_scale)
            z_scale = z_log_scale * 0.1  # start small; softplus already applied in head
            if z_mean.ndim == 3:
                z_scale = z_scale.unsqueeze(1).expand_as(z_mean)
            z_mean = z_mean + z_scale * torch.randn_like(z_mean)

        return z_mean

    def compute_loss(
        self,
        state_t: torch.Tensor,
        state_t1: torch.Tensor,
        market_cond: torch.Tensor,
    ) -> LeWMLossOutput:
        """Compute full LeWM training loss with return MSE + volatility matching.

        Without decoder:
            L_total = L_pred + L_reg + lambda_price * L_ret_mse + lambda_returns * L_vol
        With decoder:
            L_total = L_pred + L_reg + lambda_recon * L_recon
                      + lambda_price * L_ret_mse + lambda_returns * L_vol

        Both states are encoded by the same encoder (end-to-end,
        no stop-gradient on the target -- following the paper).

        Args:
            state_t: Current agent grid [B, H, W, d_agent].
            state_t1: Next agent grid [B, H, W, d_agent].
            market_cond: Market conditions [B, d_cond].

        Returns:
            LeWMLossOutput with all loss components and latent vectors.
        """
        # Encode both states (no stop-gradient, no EMA)
        z_t = self.encoder(state_t)       # [B, d_latent]
        z_t1 = self.encoder(state_t1)     # [B, d_latent]

        # Predict next latent (with stochastic noise during training)
        z_t1_pred = self.predict(z_t, market_cond, stochastic=True)  # [B, d_latent]

        # Prediction loss: MSE in latent space
        loss_pred = F.mse_loss(z_t1_pred, z_t1)

        # --- Anti-collapse regularization (VICReg-style + SIGReg) ---
        z_all = torch.cat([z_t, z_t1], dim=0)  # [2B, d_latent]

        # 1) Variance: per-dim std must be >= 1 (hinge loss)
        z_std = z_all.std(dim=0)  # [d_latent]
        loss_var = F.relu(1.0 - z_std).mean()

        # 2) Covariance: off-diagonal of cov matrix should be zero
        z_centered = z_all - z_all.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(z_all.shape[0] - 1, 1)
        off_diag = cov - torch.diag(cov.diag())
        loss_cov = (off_diag.pow(2)).mean()

        # 3) SIGReg (original)
        loss_sigreg = self.sigreg(z_all)

        # Combined regularization (reduced from original 25/1/5 to prevent
        # over-regularization that dilutes prediction signal)
        loss_reg = loss_var * 2.0 + loss_cov * 0.1 + loss_sigreg * self.lambda_sigreg

        # Total loss (before decoder, returns)
        loss_total = loss_pred + loss_reg

        # --- Reconstruction loss (decoder) ---
        loss_recon: torch.Tensor | None = None
        if self.decoder is not None:
            # Decode both z_t and z_t1, compare to original states in raw space
            state_t_recon = self.decoder(z_t)    # [B, H, W, d_agent]
            state_t1_recon = self.decoder(z_t1)  # [B, H, W, d_agent]
            loss_recon = 0.5 * (
                F.mse_loss(state_t_recon, state_t)
                + F.mse_loss(state_t1_recon, state_t1)
            )
            loss_total = loss_total + self.lambda_recon * loss_recon

        # --- Return MSE + volatility matching loss ---
        # Adaptive: use dim 98 for ABIDES (128-dim), dim 0 for crypto (32-dim)
        d_agent = state_t.shape[-1]
        if d_agent > 64:
            # ABIDES mode: dim 98 = normalized mid price
            price_dim = 98
            mask_hw = (state_t[..., 0] != 0) | (state_t[..., 1] != 0)
            price_t_gt = masked_mean(state_t[..., price_dim], mask_hw, dims=(1, 2))
            price_t1_gt = masked_mean(state_t1[..., price_dim], mask_hw, dims=(1, 2))
            ret_gt = (torch.log(price_t1_gt.clamp(min=0.01))
                      - torch.log(price_t_gt.clamp(min=0.01)))
        else:
            # Crypto mode: dim 0 = normalized log_return, use mean across window
            B = state_t.shape[0]
            ret_t = state_t.reshape(B, -1, d_agent)[:, :, 0].mean(dim=1)
            ret_t1 = state_t1.reshape(B, -1, d_agent)[:, :, 0].mean(dim=1)
            ret_gt = ret_t1 - ret_t

        # Data-std based sigma floor for better calibration
        data_std = ret_gt.detach().std().clamp(min=1e-6)
        sigma_floor = (data_std * 0.5).item()

        # Predict return distribution parameters from predicted latent
        mu, sigma, nu = self.return_dist_head(z_t1_pred, sigma_floor=sigma_floor)

        # Asymmetric volatility (leverage effect):
        # sigma increases after negative returns
        ret_prev = market_cond[:, 5]  # previous return from market_cond (dim 5 = log_return at t)
        neg_shock = F.relu(-ret_prev)
        sigma = sigma * (1.0 + self.beta_leverage * neg_shock)

        # MSE on predicted vs actual returns
        loss_ret_mse = F.mse_loss(mu, ret_gt)

        # Volatility matching: MSE between predicted sigma and realised |return|
        loss_vol = F.mse_loss(sigma, ret_gt.detach().abs())

        # lambda_price -> weight for return MSE; lambda_returns -> weight for vol matching
        loss_total = (loss_total
                      + self.lambda_price * loss_ret_mse
                      + self.lambda_returns * loss_vol)

        return LeWMLossOutput(
            loss_total=loss_total,
            loss_pred=loss_pred,
            loss_sigreg=loss_reg,  # report combined reg loss
            loss_recon=loss_recon,
            loss_price=loss_ret_mse,     # repurposed field
            loss_returns=loss_vol,       # repurposed field
            loss_ret_mse=loss_ret_mse,
            loss_vol=loss_vol,
            z_t=z_t,
            z_t1_target=z_t1,
            z_t1_pred=z_t1_pred,
        )

    def forward(
        self,
        state_t: torch.Tensor,
        market_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Inference mode: encode state and predict next latent (deterministic).

        Args:
            state_t: Current agent grid [B, H, W, d_agent].
            market_cond: Market conditions [B, d_cond].

        Returns:
            z_t1_pred: Predicted next-step latent [B, d_latent].
        """
        z_t = self.encoder(state_t)
        return self.predict(z_t, market_cond, stochastic=False)


def build_lewm(cfg) -> LeWorldModel:
    """Build LeWorldModel from config object.

    Expected config structure (lewm section):
        lewm:
            d_enc, d_latent, d_pred, d_cond,
            enc_depth, enc_heads, pred_depth, pred_heads,
            enc_mlp_ratio, pred_mlp_ratio,
            num_projections, lambda_sigreg, lambda_recon,
            dropout, use_decoder,
            d_dec, dec_depth, dec_heads, dec_mlp_ratio
    """
    lc = cfg.lewm
    p = cfg.patch.patch_size
    # Compute padded grid dims (round up to nearest multiple of patch_size)
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
        lambda_sigreg=lc.get("lambda_sigreg", 0.1),
        lambda_recon=lc.get("lambda_recon", 1.0),
        lambda_price=lc.get("lambda_price", 10.0),
        lambda_returns=lc.get("lambda_returns", 5.0),
        beta_leverage=lc.get("beta_leverage", 0.5),
        dropout=lc.get("dropout", 0.0),
        use_decoder=lc.get("use_decoder", False),
        d_dec=lc.get("d_dec", 256),
        dec_depth=lc.get("dec_depth", 4),
        dec_heads=lc.get("dec_heads", 4),
        dec_mlp_ratio=lc.get("dec_mlp_ratio", 4.0),
        dec_grid_h=pad_h,
        dec_grid_w=pad_w,
    )
