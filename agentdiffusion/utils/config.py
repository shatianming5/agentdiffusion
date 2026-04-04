"""Configuration loading utilities based on OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig


# ---------------------------------------------------------------------------
# Dataclass-based config hierarchy
# ---------------------------------------------------------------------------

@dataclass
class AgentStateConfig:
    raw_dim: int = 128
    latent_dim: int = 16
    agent_types: int = 4  # market_maker, trend_follower, fundamentalist, noise_trader


@dataclass
class PatchConfig:
    patch_size: int = 4
    grid_h: int = 100
    grid_w: int = 100


@dataclass
class ModelConfig:
    d_model: int = 512
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    num_market_tokens: int = 128
    local_window_size: int = 8
    market_cond_dim: int = 32
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    prediction_type: str = "v_prediction"  # "epsilon" | "x0" | "v_prediction"
    ddim_steps: int = 50


@dataclass
class ConstraintConfig:
    lambda_clearing: float = 1.0
    lambda_budget: float = 0.5
    lambda_conservation: float = 1.0
    guidance_scale_clearing: float = 2.0
    guidance_scale_budget: float = 1.0
    guidance_start_ratio: float = 0.5  # 仅在后半段去噪开启 guidance


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5000
    total_steps: int = 500_000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"
    log_every: int = 100
    save_every: int = 10_000
    eval_every: int = 5_000
    rollout_steps: int = 1   # 1 = single-step (backward compat), >1 = multi-step rollout
    seq_len: int = 8         # sequence length for AgentSequenceDataset


@dataclass
class DataConfig:
    data_dir: str = "data/abides_small"
    num_workers: int = 4
    pin_memory: bool = True
    # Crypto-specific fields (ignored by ABIDES pipeline)
    symbol: str = "BTCUSDT"
    stride: int = 1


@dataclass
class VideoConfig:
    """Video DiT sequence configuration."""
    num_frames: int = 40          # Total frames per sequence (K + N)
    num_cond_frames: int = 8      # K condition frames (clean)
    # num_gen_frames = num_frames - num_cond_frames = 32


@dataclass
class LeWMConfig:
    """LeWorldModel (JEPA) architecture configuration."""
    d_enc: int = 256
    d_latent: int = 256
    d_pred: int = 384
    d_cond: int = 32
    enc_depth: int = 6
    enc_heads: int = 8
    pred_depth: int = 6
    pred_heads: int = 8
    enc_mlp_ratio: float = 4.0
    pred_mlp_ratio: float = 2.0
    dropout: float = 0.0
    num_projections: int = 512
    lambda_sigreg: float = 0.5
    # Price / returns supervision
    lambda_price: float = 1.0
    lambda_returns: float = 1.0
    # Decoder
    use_decoder: bool = True
    d_dec: int = 256
    dec_depth: int = 4
    dec_heads: int = 4
    dec_mlp_ratio: float = 4.0
    lambda_recon: float = 1.0


@dataclass
class AgentDiffusionConfig:
    agent: AgentStateConfig = field(default_factory=AgentStateConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    lewm: LeWMConfig = field(default_factory=LeWMConfig)
    seed: int = 42
    output_dir: str = "outputs"


def load_config(path: str | None = None, overrides: list[str] | None = None) -> AgentDiffusionConfig:
    """Load config from YAML, merge with defaults and CLI overrides."""
    base = OmegaConf.structured(AgentDiffusionConfig)
    if path is not None:
        file_cfg = OmegaConf.load(path)
        base = OmegaConf.merge(base, file_cfg)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, cli_cfg)
    OmegaConf.resolve(base)
    return base  # type: ignore[return-value]
