"""Video dataset: loads consecutive sequences of agent grid frames for Video DiT training.

Each simulation produces a series of transition .pt files with keys:
    state_t:     [H, W, C]       -- agent grid at timestep t
    state_t1:    [H, W, C]       -- agent grid at timestep t+1
    market_cond: [cond_dim]      -- market condition vector
    sim_id:      int             -- simulation run identifier
    time_index:  int             -- timestep index within the simulation

The dataset groups files by sim_id, sorts by time_index, and constructs
sliding windows of `total_frames` consecutive grid states.  Each window
yields 40 frames (8 condition + 32 generation) that the Video DiT
processes as a single training sample.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _pad_grid(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Pad [H, W, ...] tensor to [target_h, target_w, ...] with zeros."""
    h, w = tensor.shape[:2]
    if h >= target_h and w >= target_w:
        return tensor[:target_h, :target_w]
    pad_h, pad_w = target_h - h, target_w - w
    if tensor.ndim == 3:
        return F.pad(tensor, (0, 0, 0, max(0, pad_w), 0, max(0, pad_h)))
    elif tensor.ndim == 2:
        return F.pad(tensor, (0, max(0, pad_w), 0, max(0, pad_h)), value=-1)
    return tensor


class AgentVideoDataset(Dataset):
    """Loads consecutive sequences of agent grid frames for video diffusion.

    Produces sliding windows of `total_frames` states from each simulation.
    Each transition file contains (state_t, state_t1), so K consecutive files
    yield K+1 unique states.  We collect enough files to produce `total_frames`
    states per window.

    Args:
        data_dir:     Directory containing .pt transition files.
        total_frames: Number of frames per sequence (default 40 = 8 cond + 32 gen).
        cond_frames:  Number of clean conditioning frames (default 8).
        pad_to:       Pad grids to (H, W) so they are patch-size divisible.
        market_cond_dim: Dimension of market condition vector (for fallback zeros).
    """

    def __init__(
        self,
        data_dir: str,
        total_frames: int = 40,
        cond_frames: int = 8,
        pad_to: tuple[int, int] = (36, 36),
        market_cond_dim: int = 32,
    ):
        self.data_dir = Path(data_dir)
        self.total_frames = total_frames
        self.cond_frames = cond_frames
        self.gen_frames = total_frames - cond_frames
        self.pad_to = pad_to
        self.market_cond_dim = market_cond_dim

        all_files = sorted(self.data_dir.glob("*.pt"))
        if not all_files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

        # Group files by sim_id, sort within each group by time_index.
        # We need (total_frames - 1) transition files to get total_frames states
        # (each file provides state_t -> state_t1, so consecutive files chain).
        num_transitions = total_frames - 1

        # Pre-load all data into memory for fast access
        # (9800 files × ~1MB = ~10GB, fits in RAM)
        self._file_cache: dict[str, dict] = {}
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Pre-loading {len(all_files)} files into memory...")
        for f in all_files:
            self._file_cache[str(f)] = torch.load(f, map_location="cpu", weights_only=True)

        # Group files by sim_id using cached data
        sim_groups: dict[int, list[tuple[int, Path]]] = {}
        for f in all_files:
            data = self._file_cache[str(f)]
            sid = int(data.get("sim_id", 0))
            tidx = int(data.get("time_index", 0))
            sim_groups.setdefault(sid, []).append((tidx, f))

        # Build sliding windows: each window is a list of consecutive file paths
        self.windows: list[list[Path]] = []
        for sid in sorted(sim_groups.keys()):
            group = sorted(sim_groups[sid], key=lambda x: x[0])
            paths = [p for _, p in group]
            if len(paths) >= num_transitions:
                for start in range(len(paths) - num_transitions + 1):
                    self.windows.append(paths[start : start + num_transitions])

        if not self.windows:
            raise ValueError(
                f"No sequences of length {total_frames} found. "
                f"Have {len(all_files)} files across {len(sim_groups)} simulations. "
                f"Need at least {num_transitions} consecutive transitions per sim."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sequence of total_frames consecutive frames + market conditions.

        Returns:
            dict with:
                "frames":       [T, H, W, C] -- all T frames (raw agent features).
                "market_conds": [T, cond_dim] -- market conditions per frame.
        """
        window = self.windows[idx]
        th, tw = self.pad_to

        frames: list[torch.Tensor] = []
        market_conds: list[torch.Tensor] = []

        for i, fpath in enumerate(window):
            data = self._file_cache[str(fpath)]
            st = data["state_t"]
            st1 = data["state_t1"]
            mc = data.get("market_cond", torch.zeros(self.market_cond_dim))

            # Pad grids
            st = _pad_grid(st, th, tw)
            st1 = _pad_grid(st1, th, tw)

            if i == 0:
                # First file: add state_t as the first frame
                frames.append(st)
                market_conds.append(mc)

            # Each subsequent file contributes state_t1
            frames.append(st1)
            market_conds.append(mc)

        # Stack: [T, H, W, C] and [T, cond_dim]
        frames_tensor = torch.stack(frames[:self.total_frames])
        conds_tensor = torch.stack(market_conds[:self.total_frames])

        return {
            "frames": frames_tensor,
            "market_conds": conds_tensor,
        }


class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset for testing (no ABIDES dependency).

    Generates random but structurally plausible frame sequences.
    Each sequence starts with a random base state and applies small
    perturbations for successive frames, simulating temporal evolution.
    """

    def __init__(
        self,
        num_samples: int = 500,
        total_frames: int = 40,
        cond_frames: int = 8,
        grid_h: int = 36,
        grid_w: int = 36,
        raw_dim: int = 128,
        market_cond_dim: int = 32,
    ):
        self.num_samples = num_samples
        self.total_frames = total_frames
        self.cond_frames = cond_frames
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.raw_dim = raw_dim
        self.market_cond_dim = market_cond_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(idx)

        # Generate a temporally correlated sequence
        base = torch.randn(self.grid_h, self.grid_w, self.raw_dim, generator=gen)
        frames = [base]
        for _ in range(self.total_frames - 1):
            delta = torch.randn_like(base, generator=gen) * 0.05
            base = base + delta
            frames.append(base.clone())

        market_conds = [
            torch.randn(self.market_cond_dim, generator=gen)
            for _ in range(self.total_frames)
        ]

        return {
            "frames": torch.stack(frames),           # [T, H, W, C]
            "market_conds": torch.stack(market_conds),  # [T, cond_dim]
        }
