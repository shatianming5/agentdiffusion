"""Agent state representation and grid layout utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as F


class AgentType(IntEnum):
    MARKET_MAKER = 0
    TREND_FOLLOWER = 1
    FUNDAMENTALIST = 2
    NOISE_TRADER = 3


# 每种 agent 类型在 128 维状态向量中的语义分段
STATE_SLICES = {
    "positions":   slice(0, 32),    # 各资产持有量
    "funds":       slice(32, 48),   # 现金、杠杆、保证金
    "strategy":    slice(48, 64),   # agent 类型编码 + 策略超参
    "history":     slice(64, 80),   # 近期收益、波动率、夏普
    "behavior":    slice(80, 96),   # 下单频率、撤单率
    "observation": slice(96, 112),  # agent 感知的价格/深度
    "social":      slice(112, 128), # 信息来源、跟风系数
}


@dataclass
class AgentGrid:
    """Utility to arrange N agents into an H×W grid by type and capital."""

    num_agents: int
    type_ratios: dict[AgentType, float] | None = None

    def __post_init__(self):
        if self.type_ratios is None:
            self.type_ratios = {
                AgentType.MARKET_MAKER: 0.02,
                AgentType.TREND_FOLLOWER: 0.30,
                AgentType.FUNDAMENTALIST: 0.20,
                AgentType.NOISE_TRADER: 0.48,
            }
        h = int(math.ceil(math.sqrt(self.num_agents)))
        w = int(math.ceil(self.num_agents / h))
        self.grid_h = h
        self.grid_w = w
        self.grid_size = h * w  # may be >= num_agents (padding)

    def arrange(
        self,
        states: torch.Tensor,   # [N, C]
        types: torch.Tensor,    # [N]  int
        capitals: torch.Tensor, # [N]  float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort agents by (type, capital) and reshape into H×W grid.

        Returns:
            grid_states:  [H, W, C]
            grid_types:   [H, W]
            sort_indices: [N]  for reversing the permutation
        """
        N, C = states.shape
        # 主排序: type 升序; 副排序: capital 降序（同 type 内大资金在前）
        sort_key = types.float() * 1e12 - capitals
        sort_indices = sort_key.argsort()

        sorted_states = states[sort_indices]
        sorted_types = types[sort_indices]

        # pad 到 grid_size
        pad_n = self.grid_size - N
        if pad_n > 0:
            sorted_states = F.pad(sorted_states, (0, 0, 0, pad_n))
            sorted_types = F.pad(sorted_types, (0, pad_n), value=-1)

        grid_states = sorted_states.view(self.grid_h, self.grid_w, C)
        grid_types = sorted_types.view(self.grid_h, self.grid_w)
        return grid_states, grid_types, sort_indices

    def flatten(
        self,
        grid_states: torch.Tensor,  # [H, W, C]
        sort_indices: torch.Tensor, # [N]
    ) -> torch.Tensor:
        """Reverse grid back to original agent order. Returns [N, C]."""
        flat = grid_states.view(-1, grid_states.shape[-1])
        N = sort_indices.shape[0]
        flat = flat[:N]
        # invert permutation
        inv = torch.empty_like(sort_indices)
        inv[sort_indices] = torch.arange(N, device=sort_indices.device)
        return flat[inv]


def normalize_states(states: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Normalize raw agent states. Returns (normalized, stats_for_denorm)."""
    stats = {}
    out = states.clone()

    # 持仓: signed log
    pos = out[:, STATE_SLICES["positions"]]
    out[:, STATE_SLICES["positions"]] = pos.sign() * torch.log1p(pos.abs())
    stats["positions_applied"] = "signed_log1p"

    # 资金: 除以均值做相对化
    funds = out[:, STATE_SLICES["funds"]]
    funds_mean = funds.abs().mean(dim=0, keepdim=True).clamp(min=1e-8)
    out[:, STATE_SLICES["funds"]] = funds / funds_mean
    stats["funds_mean"] = funds_mean

    # 其余维度: z-score
    for name in ("history", "behavior", "observation", "social"):
        s = STATE_SLICES[name]
        block = out[:, s]
        mu = block.mean(dim=0, keepdim=True)
        sigma = block.std(dim=0, keepdim=True).clamp(min=1e-8)
        out[:, s] = (block - mu) / sigma
        stats[f"{name}_mu"] = mu
        stats[f"{name}_sigma"] = sigma

    return out, stats


def denormalize_states(normed: torch.Tensor, stats: dict) -> torch.Tensor:
    """Inverse of normalize_states."""
    out = normed.clone()

    # 持仓: inverse signed log
    pos = out[:, STATE_SLICES["positions"]]
    out[:, STATE_SLICES["positions"]] = pos.sign() * (torch.exp(pos.abs()) - 1)

    # 资金
    out[:, STATE_SLICES["funds"]] = out[:, STATE_SLICES["funds"]] * stats["funds_mean"]

    # z-score 还原
    for name in ("history", "behavior", "observation", "social"):
        s = STATE_SLICES[name]
        out[:, s] = out[:, s] * stats[f"{name}_sigma"] + stats[f"{name}_mu"]

    return out
