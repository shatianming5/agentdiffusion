"""PyTorch Dataset for agent state transitions."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _pad_grid(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Pad [H, W, ...] tensor to [target_h, target_w, ...] with zeros."""
    h, w = tensor.shape[:2]
    if h >= target_h and w >= target_w:
        return tensor[:target_h, :target_w]
    pad_h, pad_w = target_h - h, target_w - w
    if tensor.ndim == 3:
        return torch.nn.functional.pad(tensor, (0, 0, 0, max(0, pad_w), 0, max(0, pad_h)))
    elif tensor.ndim == 2:
        return torch.nn.functional.pad(tensor, (0, max(0, pad_w), 0, max(0, pad_h)), value=-1)
    return tensor


class AgentTransitionDataset(Dataset):
    """Dataset of (state_t, state_t1, market_cond) tuples.

    Each .pt file in data_dir contains a dict with keys:
        state_t:     [H, W, C]
        state_t1:    [H, W, C]
        market_cond: [market_cond_dim]
        agent_types: [H, W]

    If pad_to is set, grids are padded to (pad_h, pad_w) so they are
    divisible by the patch size used in the diffusion model.
    """

    def __init__(self, data_dir: str, transform=None, pad_to: tuple[int, int] | None = None):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        self.transform = transform
        self.pad_to = pad_to

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        if self.pad_to is not None:
            th, tw = self.pad_to
            data["state_t"] = _pad_grid(data["state_t"], th, tw)
            data["state_t1"] = _pad_grid(data["state_t1"], th, tw)
            if "agent_types" in data:
                data["agent_types"] = _pad_grid(data["agent_types"], th, tw)
        if self.transform:
            data = self.transform(data)
        return data


class AgentFlatDataset(Dataset):
    """Dataset for autoencoder pretraining: individual agent state vectors.

    Loads full grid files and indexes into individual agents.
    """

    def __init__(self, data_dir: str, grid_h: int, grid_w: int):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        self.agents_per_file = grid_h * grid_w
        self.total = len(self.files) * self.agents_per_file
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Cache: lazy load
        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.total

    def _load_file(self, file_idx: int) -> torch.Tensor:
        if file_idx not in self._cache:
            data = torch.load(self.files[file_idx], map_location="cpu", weights_only=True)
            states = data["state_t"]  # [H, W, C]
            self._cache[file_idx] = states.reshape(-1, states.shape[-1])
            # Keep cache bounded — never evict the key we just inserted
            if len(self._cache) > 50:
                evict = min(k for k in self._cache if k != file_idx)
                del self._cache[evict]
        return self._cache[file_idx]

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = idx // self.agents_per_file
        agent_idx = idx % self.agents_per_file
        agents = self._load_file(file_idx)
        return agents[agent_idx]  # [C]


class AgentSequenceDataset(Dataset):
    """Dataset that returns sequences of K consecutive transitions from same simulation.

    Groups .pt files by sim_id, sorts within each group by time_index,
    then yields sliding windows of seq_len consecutive (state, market_cond) pairs.

    Each __getitem__ returns a dict:
        seq_states:    [K+1, H, W, C]  — K+1 states covering K transitions
        seq_conds:     [K, cond_dim]   — market conditions for each transition
        seq_agent_types: [H, W]        — agent types (from first timestep)
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 4,
        pad_to: tuple[int, int] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.pad_to = pad_to

        all_files = sorted(self.data_dir.glob("*.pt"))
        if not all_files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

        # Group files by sim_id, sort by time_index within each group
        sim_groups: dict[int, list[tuple[int, Path]]] = {}
        for f in all_files:
            data = torch.load(f, map_location="cpu", weights_only=True)
            sid = int(data.get("sim_id", 0))
            tidx = int(data.get("time_index", 0))
            sim_groups.setdefault(sid, []).append((tidx, f))

        # Sort each group by time_index and build sliding windows
        self.windows: list[list[Path]] = []
        for sid in sorted(sim_groups.keys()):
            group = sorted(sim_groups[sid], key=lambda x: x[0])
            paths = [p for _, p in group]
            # Each transition file already contains (state_t, state_t1).
            # A window of seq_len transitions needs seq_len consecutive files.
            if len(paths) >= seq_len:
                for start in range(len(paths) - seq_len + 1):
                    self.windows.append(paths[start : start + seq_len])

        if not self.windows:
            raise ValueError(
                f"No sequences of length {seq_len} found. "
                f"Have {len(all_files)} files across {len(sim_groups)} sims."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = self.windows[idx]
        states: list[torch.Tensor] = []
        conds: list[torch.Tensor] = []
        agent_types = None

        for i, fpath in enumerate(window):
            data = torch.load(fpath, map_location="cpu", weights_only=True)
            st = data["state_t"]
            st1 = data["state_t1"]
            mc = data["market_cond"]

            if self.pad_to is not None:
                th, tw = self.pad_to
                st = _pad_grid(st, th, tw)
                st1 = _pad_grid(st1, th, tw)

            if i == 0:
                states.append(st)
                if "agent_types" in data:
                    at = data["agent_types"]
                    if self.pad_to is not None:
                        at = _pad_grid(at, self.pad_to[0], self.pad_to[1])
                    agent_types = at
            states.append(st1)
            conds.append(mc)

        result: dict[str, torch.Tensor] = {
            "seq_states": torch.stack(states),   # [K+1, H, W, C]
            "seq_conds": torch.stack(conds),     # [K, cond_dim]
        }
        if agent_types is not None:
            result["seq_agent_types"] = agent_types
        return result


class SyntheticAgentDataset(Dataset):
    """Generates synthetic agent data for testing (no ABIDES dependency).

    Creates random but structurally valid agent state transitions.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        grid_h: int = 32,
        grid_w: int = 32,
        raw_dim: int = 128,
        market_cond_dim: int = 32,
        num_agent_types: int = 4,
    ):
        self.num_samples = num_samples
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.raw_dim = raw_dim
        self.market_cond_dim = market_cond_dim
        self.num_agent_types = num_agent_types

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(idx)

        state_t = torch.randn(self.grid_h, self.grid_w, self.raw_dim, generator=gen)
        # Next state = current + small perturbation (simulates one-step dynamics)
        delta = torch.randn_like(state_t, generator=gen) * 0.1
        state_t1 = state_t + delta

        market_cond = torch.randn(self.market_cond_dim, generator=gen)
        agent_types = torch.randint(
            0, self.num_agent_types, (self.grid_h, self.grid_w), generator=gen
        )

        return {
            "state_t": state_t,
            "state_t1": state_t1,
            "market_cond": market_cond,
            "agent_types": agent_types,
        }
