"""Masked aggregation utilities shared across LeWM and Diffusion pipelines."""

from __future__ import annotations

import torch


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]
) -> torch.Tensor:
    """Average tensor over dims, only counting positions where mask is True.

    Args:
        tensor: Values to average, arbitrary shape.
        mask: Boolean mask, broadcastable to tensor shape.
        dims: Dimensions over which to reduce.

    Returns:
        Reduced tensor with dims collapsed.
    """
    tensor = tensor * mask.float()
    return tensor.sum(dim=dims) / mask.float().sum(dim=dims).clamp(min=1)
