"""Stochastic volatility head and skewed Student-t utilities.

This module is designed to sit on top of a world model latent state z_t and
predict a heavy-tailed, asymmetric return distribution together with a volume
proxy. The head maintains a multi-scale latent volatility state with fixed
decay factors, making it easy to use in teacher-forced training or
autoregressive rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


__all__ = [
    "SkewedStudentT",
    "StochasticVolatilityHead",
    "StochasticVolatilityOutput",
    "StochasticVolatilitySequenceOutput",
    "TrainStepOutput",
    "acf_whiteness_penalty",
    "leverage_moment_penalty",
    "train_step",
]


@dataclass
class StochasticVolatilityOutput:
    """Single-step stochastic volatility head output."""

    mu: torch.Tensor
    sigma: torch.Tensor
    nu: torch.Tensor
    skew: torch.Tensor
    log_volume: torch.Tensor
    h_next: torch.Tensor
    eps: torch.Tensor
    scale_features: torch.Tensor
    scale_weights: torch.Tensor


@dataclass
class StochasticVolatilitySequenceOutput:
    """Teacher-forced rollout output over T steps."""

    mu: torch.Tensor
    sigma: torch.Tensor
    nu: torch.Tensor
    skew: torch.Tensor
    log_volume: torch.Tensor
    eps: torch.Tensor
    h_final: torch.Tensor


@dataclass
class TrainStepOutput:
    """Container returned by train_step."""

    loss: torch.Tensor
    nll: torch.Tensor
    mu_penalty: torch.Tensor
    acf_penalty: torch.Tensor
    leverage_penalty: torch.Tensor
    volume_loss: torch.Tensor
    rollout: StochasticVolatilitySequenceOutput
    standardized_residuals: torch.Tensor


class SkewedStudentT(Distribution):
    """Fernandez-Steel skewed Student-t distribution.

    The distribution is defined by skewing a standard Student-t density
    with the Fernandez-Steel transformation:

        s(x; gamma) = 2 / (gamma + 1 / gamma) *
            [f(x / gamma) I[x >= 0] + f(gamma x) I[x < 0]]

    where f is the standard Student-t density and gamma > 0 controls
    left/right scaling. The public skew parameter is bounded in [-1, 1)
    and internally mapped to gamma.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "df": constraints.greater_than(2.0),
        "skew": constraints.interval(-0.999, 0.999),
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        df: torch.Tensor,
        skew: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        loc, scale, df, skew = broadcast_all(loc, scale, df, skew)
        self.loc = loc
        self.scale = scale.clamp(min=1e-6)  # ensure strictly positive
        self.df = df.clamp(min=2.01)
        self.skew = skew.clamp(min=-0.998, max=0.998)
        batch_shape = self.loc.size()
        super().__init__(batch_shape=batch_shape, validate_args=False)

    @staticmethod
    def _skew_to_gamma(skew: torch.Tensor) -> torch.Tensor:
        clipped = skew.clamp(min=-0.999, max=0.999)
        return torch.sqrt((1.0 + clipped) / (1.0 - clipped))

    @staticmethod
    def _base_log_prob(x: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        half_df = 0.5 * df
        return (
            torch.lgamma(half_df + 0.5)
            - torch.lgamma(half_df)
            - 0.5 * (torch.log(df) + math.log(math.pi))
            - (half_df + 0.5) * torch.log1p(x.pow(2) / df)
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        y = (value - self.loc) / self.scale
        gamma = self._skew_to_gamma(self.skew)
        inv_gamma = gamma.reciprocal()
        base_arg = torch.where(y >= 0.0, y / gamma, y * gamma)
        log_norm = math.log(2.0) - torch.log(gamma + inv_gamma) - torch.log(self.scale)
        return log_norm + self._base_log_prob(base_arg, self.df)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        loc = self.loc.expand(shape)
        scale = self.scale.expand(shape)
        df = self.df.expand(shape)
        gamma = self._skew_to_gamma(self.skew).expand(shape)

        normal = torch.randn(shape, device=loc.device, dtype=loc.dtype)
        chi2 = torch.distributions.Gamma(0.5 * df, 0.5).rsample()
        student = normal * torch.sqrt(df / chi2)

        abs_student = student.abs()
        p_pos = gamma.pow(2) / (1.0 + gamma.pow(2))
        draws = torch.rand(shape, device=loc.device, dtype=loc.dtype)
        skewed = torch.where(draws < p_pos, abs_student * gamma, -abs_student / gamma)
        return loc + scale * skewed


class StochasticVolatilityHead(nn.Module):
    """Multi-scale stochastic volatility head over world model latents.

    Inputs:
        z_t: [B, d_latent]
        prev_return: [B]
        prev_sigma: [B]
        h_t: [B, num_scales * d_h] or [B, num_scales, d_h]

    Outputs:
        mu, sigma, nu, skew, log_volume, updated hidden state.
    """

    def __init__(
        self,
        d_latent: int = 256,
        d_h: int = 32,
        hidden_dim: int = 256,
        num_scales: int = 3,
        rho: Sequence[float] = (0.90, 0.97, 0.995),
        min_sigma: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_scales != len(rho):
            raise ValueError(
                f"num_scales={num_scales} must match len(rho)={len(rho)}."
            )

        self.d_latent = d_latent
        self.d_h = d_h
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.min_sigma = float(min_sigma)

        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))

        feature_dim = 5  # prev_return, log_prev_sigma, eps, |eps|, relu(-eps)

        self.innovation_net = nn.Sequential(
            nn.Linear(d_latent + feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_scales * d_h),
        )
        self.scale_summary = nn.Linear(d_h, 1)
        self.scale_gate = nn.Sequential(
            nn.Linear(d_latent + feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_scales),
        )
        self.context_net = nn.Sequential(
            nn.Linear(d_latent + num_scales * d_h + feature_dim + num_scales, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.param_head = nn.Linear(hidden_dim, 4)
        self.volume_head = nn.Linear(hidden_dim, 1)
        self.state_norm = nn.LayerNorm(num_scales * d_h)

        nn.init.zeros_(self.param_head.bias)
        nn.init.zeros_(self.volume_head.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return a zero-initialized hidden volatility state."""

        return torch.zeros(
            batch_size,
            self.num_scales * self.d_h,
            device=device,
            dtype=dtype,
        )

    def _reshape_state(
        self,
        h_t: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if h_t is None:
            return self.initial_state(batch_size, device=device, dtype=dtype)
        if h_t.ndim == 2:
            expected = self.num_scales * self.d_h
            if h_t.shape != (batch_size, expected):
                raise ValueError(
                    f"h_t must have shape [B, {expected}], got {tuple(h_t.shape)}."
                )
            return h_t
        if h_t.ndim == 3:
            if h_t.shape != (batch_size, self.num_scales, self.d_h):
                raise ValueError(
                    "h_t must have shape [B, num_scales, d_h] when 3D, "
                    f"got {tuple(h_t.shape)}."
                )
            return h_t.reshape(batch_size, -1)
        raise ValueError(
            f"h_t must be None, [B, {self.num_scales * self.d_h}] or "
            f"[B, {self.num_scales}, {self.d_h}]."
        )

    def forward(
        self,
        z_t: torch.Tensor,
        prev_return: torch.Tensor,
        prev_sigma: torch.Tensor,
        h_t: torch.Tensor | None = None,
    ) -> StochasticVolatilityOutput:
        if z_t.ndim != 2 or z_t.shape[-1] != self.d_latent:
            raise ValueError(
                f"z_t must have shape [B, {self.d_latent}], got {tuple(z_t.shape)}."
            )

        batch_size = z_t.shape[0]
        flat_state = self._reshape_state(h_t, batch_size, z_t.device, z_t.dtype)
        prev_return = prev_return.reshape(batch_size, 1).to(device=z_t.device, dtype=z_t.dtype)
        prev_sigma = (
            prev_sigma.reshape(batch_size, 1)
            .to(device=z_t.device, dtype=z_t.dtype)
            .clamp_min(self.min_sigma)
        )

        eps = prev_return / prev_sigma
        shock_features = torch.cat(
            [
                prev_return,
                prev_sigma.log(),
                eps,
                eps.abs(),
                F.relu(-eps),
            ],
            dim=-1,
        )

        innovation_in = torch.cat([z_t, shock_features], dim=-1)
        innovation = self.innovation_net(innovation_in).view(
            batch_size, self.num_scales, self.d_h
        )

        h_prev = flat_state.view(batch_size, self.num_scales, self.d_h)
        rho = self.rho.to(device=z_t.device, dtype=z_t.dtype).view(1, self.num_scales, 1)
        h_next = rho * h_prev + (1.0 - rho) * innovation
        h_next_flat = h_next.reshape(batch_size, -1)

        scale_features = self.scale_summary(h_next).squeeze(-1)
        scale_weights = torch.softmax(self.scale_gate(innovation_in), dim=-1)

        context = torch.cat(
            [
                z_t,
                self.state_norm(h_next_flat),
                shock_features,
                scale_features,
            ],
            dim=-1,
        )
        hidden = self.context_net(context)
        raw_mu, raw_sigma, raw_nu, raw_skew = self.param_head(hidden).unbind(dim=-1)

        mu = torch.tanh(raw_mu) * 1e-3
        log_sigma_step = 0.5 * torch.tanh(raw_sigma + (scale_weights * scale_features).sum(dim=-1))
        sigma = torch.exp(prev_sigma.squeeze(-1).log() + log_sigma_step).clamp_min(self.min_sigma)
        nu = 2.1 + (5.0 - 2.1) * torch.sigmoid(raw_nu)
        skew = 0.8 * torch.tanh(raw_skew)
        log_volume = self.volume_head(hidden).squeeze(-1)

        return StochasticVolatilityOutput(
            mu=mu,
            sigma=sigma,
            nu=nu,
            skew=skew,
            log_volume=log_volume,
            h_next=h_next_flat,
            eps=eps.squeeze(-1),
            scale_features=scale_features,
            scale_weights=scale_weights,
        )

    def teacher_forced_rollout(
        self,
        z_seq: torch.Tensor,
        returns: torch.Tensor,
        prev_return: torch.Tensor,
        prev_sigma: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> StochasticVolatilitySequenceOutput:
        """Roll the head forward over T steps with teacher-forced returns."""

        if z_seq.ndim != 3:
            raise ValueError(f"z_seq must have shape [B, T, d_latent], got {tuple(z_seq.shape)}.")
        if returns.ndim != 2:
            raise ValueError(f"returns must have shape [B, T], got {tuple(returns.shape)}.")
        if z_seq.shape[:2] != returns.shape:
            raise ValueError(
                "z_seq and returns must agree on [B, T], got "
                f"{tuple(z_seq.shape[:2])} and {tuple(returns.shape)}."
            )

        batch_size, horizon, _ = z_seq.shape
        h_t = self._reshape_state(h0, batch_size, z_seq.device, z_seq.dtype)
        prev_r = prev_return.to(device=z_seq.device, dtype=z_seq.dtype)
        prev_s = prev_sigma.to(device=z_seq.device, dtype=z_seq.dtype).clamp_min(self.min_sigma)

        mu_seq: list[torch.Tensor] = []
        sigma_seq: list[torch.Tensor] = []
        nu_seq: list[torch.Tensor] = []
        skew_seq: list[torch.Tensor] = []
        log_v_seq: list[torch.Tensor] = []
        eps_seq: list[torch.Tensor] = []

        for t in range(horizon):
            out = self(
                z_t=z_seq[:, t],
                prev_return=prev_r,
                prev_sigma=prev_s,
                h_t=h_t,
            )
            mu_seq.append(out.mu)
            sigma_seq.append(out.sigma)
            nu_seq.append(out.nu)
            skew_seq.append(out.skew)
            log_v_seq.append(out.log_volume)
            eps_seq.append(out.eps)

            h_t = out.h_next
            prev_r = returns[:, t]
            prev_s = out.sigma

        return StochasticVolatilitySequenceOutput(
            mu=torch.stack(mu_seq, dim=1),
            sigma=torch.stack(sigma_seq, dim=1),
            nu=torch.stack(nu_seq, dim=1),
            skew=torch.stack(skew_seq, dim=1),
            log_volume=torch.stack(log_v_seq, dim=1),
            eps=torch.stack(eps_seq, dim=1),
            h_final=h_t,
        )


def acf_whiteness_penalty(
    standardized_residuals: torch.Tensor,
    lags: Sequence[int] = (1, 2, 5, 10),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize autocorrelation in standardized residuals."""

    if standardized_residuals.ndim != 2:
        raise ValueError(
            "standardized_residuals must have shape [B, T], got "
            f"{tuple(standardized_residuals.shape)}."
        )

    batch_size, horizon = standardized_residuals.shape
    if batch_size == 0 or horizon == 0:
        return standardized_residuals.new_zeros(())

    centered = standardized_residuals - standardized_residuals.mean(dim=1, keepdim=True)
    denom = centered.pow(2).mean(dim=1).clamp_min(eps)

    penalties: list[torch.Tensor] = []
    for lag in lags:
        if lag <= 0 or lag >= horizon:
            continue
        acf = (centered[:, lag:] * centered[:, :-lag]).mean(dim=1) / denom
        penalties.append(acf.pow(2).mean())

    if not penalties:
        return standardized_residuals.new_zeros(())
    return torch.stack(penalties).mean()


def leverage_moment_penalty(
    prev_eps: torch.Tensor,
    sigma: torch.Tensor,
    returns: torch.Tensor,
    mu: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match leverage-sensitive volatility moments.

    Negative previous shocks should increase next-step conditional variance.
    The loss matches the weighted average predicted variance against the
    weighted average realized squared return.
    """

    if prev_eps.shape != sigma.shape or sigma.shape != returns.shape or returns.shape != mu.shape:
        raise ValueError("prev_eps, sigma, returns, and mu must all have shape [B, T].")

    neg_shock = F.relu(-prev_eps)
    weights = neg_shock / neg_shock.sum().clamp_min(eps)
    pred_moment = (weights * sigma.pow(2)).sum()
    realized_var = (returns - mu.detach()).pow(2)
    target_moment = (weights * realized_var).sum()
    return F.mse_loss(pred_moment, target_moment)


def train_step(
    model: StochasticVolatilityHead,
    optimizer: torch.optim.Optimizer | None,
    z_seq: torch.Tensor,
    returns: torch.Tensor,
    prev_return: torch.Tensor,
    prev_sigma: torch.Tensor,
    h0: torch.Tensor | None = None,
    log_volume_targets: torch.Tensor | None = None,
    *,
    nll_weight: float = 1.0,
    mu_penalty_weight: float = 1e4,
    acf_weight: float = 1.0,
    leverage_weight: float = 1.0,
    volume_weight: float = 0.0,
    acf_lags: Sequence[int] = (1, 2, 5, 10),
    grad_clip: float | None = None,
) -> TrainStepOutput:
    """Teacher-forced training step for stochastic volatility modeling.

    Args:
        model: StochasticVolatilityHead instance.
        optimizer: Optimizer to update model parameters. If None, the function
            only computes the forward pass and losses.
        z_seq: Latent sequence [B, T, d_latent].
        returns: Ground-truth returns [B, T].
        prev_return: Return at t - 1 for the first step [B].
        prev_sigma: Volatility estimate at t - 1 for the first step [B].
        h0: Optional initial hidden volatility state.
        log_volume_targets: Optional log-volume targets [B, T].
        nll_weight: Weight on skewed-t negative log likelihood.
        mu_penalty_weight: Shrinkage penalty on mu.
        acf_weight: Weight on ACF whiteness penalty.
        leverage_weight: Weight on leverage moment penalty.
        volume_weight: Optional weight on volume regression loss.
        acf_lags: Lags used for the whiteness penalty.
        grad_clip: Optional gradient clipping norm.
    """

    model.train()
    rollout = model.teacher_forced_rollout(
        z_seq=z_seq,
        returns=returns,
        prev_return=prev_return,
        prev_sigma=prev_sigma,
        h0=h0,
    )

    dist = SkewedStudentT(
        loc=rollout.mu,
        scale=rollout.sigma,
        df=rollout.nu,
        skew=rollout.skew,
    )
    nll = -dist.log_prob(returns).mean()
    mu_penalty = rollout.mu.pow(2).mean()
    standardized_residuals = (returns - rollout.mu) / rollout.sigma.clamp_min(model.min_sigma)
    acf_pen = acf_whiteness_penalty(standardized_residuals, lags=acf_lags)
    leverage_pen = leverage_moment_penalty(
        prev_eps=rollout.eps,
        sigma=rollout.sigma,
        returns=returns,
        mu=rollout.mu,
    )

    volume_loss = returns.new_zeros(())
    if log_volume_targets is not None:
        if log_volume_targets.shape != returns.shape:
            raise ValueError(
                "log_volume_targets must have shape [B, T], got "
                f"{tuple(log_volume_targets.shape)}."
            )
        volume_loss = F.mse_loss(rollout.log_volume, log_volume_targets)

    loss = (
        nll_weight * nll
        + mu_penalty_weight * mu_penalty
        + acf_weight * acf_pen
        + leverage_weight * leverage_pen
        + volume_weight * volume_loss
    )

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return TrainStepOutput(
        loss=loss,
        nll=nll,
        mu_penalty=mu_penalty,
        acf_penalty=acf_pen,
        leverage_penalty=leverage_pen,
        volume_loss=volume_loss,
        rollout=rollout,
        standardized_residuals=standardized_residuals,
    )
