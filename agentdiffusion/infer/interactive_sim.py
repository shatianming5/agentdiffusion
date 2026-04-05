"""Interactive sliding-window simulation with Video DiT.

Enables step-by-step market simulation where external interventions
(e.g., RL agent actions, shock events) can be injected between generation rounds.

Usage:
    sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16)
    sim.init(initial_frames)          # [K, H, W, d_state]

    for step in range(100):
        frames = sim.step()           # generate next N frames
        sim.intervene(frame_idx=0, mask=mm_mask, delta=shock)  # inject event
        # or: sim.set_frame(0, new_frame)  # replace a frame entirely
"""

from __future__ import annotations

import torch
from ..models.video_dit import VideoDiT, VideoDDIMSampler


class InteractiveSimulator:
    """Sliding-window interactive simulation environment.

    Maintains a rolling buffer of agent state frames. Each call to step()
    generates new frames conditioned on the tail of the buffer, then
    shifts the window forward.
    """

    def __init__(
        self,
        model: VideoDiT,
        sampler: VideoDDIMSampler,
        num_cond: int = 4,
        num_gen: int = 16,
        zero_sum_proj: bool = True,
        valid_mask: torch.Tensor | None = None,
    ):
        self.model = model
        self.sampler = sampler
        self.num_cond = num_cond
        self.num_gen = num_gen
        self.zero_sum_proj = zero_sum_proj
        self.valid_mask = valid_mask

        self.device = next(model.parameters()).device
        self.buffer: torch.Tensor | None = None  # [1, T_buf, H, W, d_state]
        self.total_steps = 0

    def init(self, initial_frames: torch.Tensor):
        """Initialize with seed frames.

        Args:
            initial_frames: [K, H, W, d_state] or [1, K, H, W, d_state]
        """
        if initial_frames.dim() == 4:
            initial_frames = initial_frames.unsqueeze(0)
        self.buffer = initial_frames.to(self.device)
        self.total_steps = 0

    @property
    def current_frames(self) -> torch.Tensor:
        """Return current buffer: [1, T_buf, H, W, d_state]."""
        return self.buffer

    @property
    def latest_frame(self) -> torch.Tensor:
        """Return the most recent frame: [H, W, d_state]."""
        return self.buffer[0, -1]

    def step(self) -> torch.Tensor:
        """Generate next num_gen frames and append to buffer.

        Returns:
            generated: [num_gen, H, W, d_state] the newly generated frames.
        """
        assert self.buffer is not None, "Call init() first"

        # Take last num_cond frames as condition
        K = self.num_cond
        x_cond = self.buffer[:, -K:]  # [1, K, H, W, d_state]

        _, _, H, W, D = x_cond.shape
        gen_shape = (1, self.num_gen, H, W, D)

        with torch.no_grad():
            generated = self.sampler.sample(
                x_cond, gen_shape, device=self.device,
                zero_sum_proj=self.zero_sum_proj,
                valid_mask=self.valid_mask,
            )  # [1, N, H, W, D]

        # Append to buffer (keep rolling)
        self.buffer = torch.cat([self.buffer, generated], dim=1)
        self.total_steps += 1

        return generated[0]  # [N, H, W, D]

    def intervene(
        self,
        frame_idx: int = -1,
        mask: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
        absolute: torch.Tensor | None = None,
    ):
        """Inject an intervention into the buffer.

        Args:
            frame_idx: which frame in the buffer to modify (-1 = last).
            mask: [H, W] bool mask of agents to affect.
            delta: [d_state] or [H, W, d_state] additive change.
            absolute: [H, W, d_state] set frame to this value (overrides delta).
        """
        assert self.buffer is not None, "Call init() first"

        if absolute is not None:
            if absolute.dim() == 3:
                absolute = absolute.unsqueeze(0)
            self.buffer[0, frame_idx] = absolute.to(self.device)
            return

        if delta is not None:
            frame = self.buffer[0, frame_idx]  # [H, W, D]
            if mask is not None:
                mask = mask.to(self.device)
                if delta.dim() == 1:
                    # Broadcast [d_state] to masked positions
                    frame[mask] = frame[mask] + delta.to(self.device)
                else:
                    frame[mask] = frame[mask] + delta[mask].to(self.device)
            else:
                if delta.dim() == 1:
                    frame = frame + delta.to(self.device)
                else:
                    frame = frame + delta.to(self.device)
                self.buffer[0, frame_idx] = frame

    def trim_buffer(self, keep_last: int | None = None):
        """Trim buffer to save memory. Keeps last N frames.

        Args:
            keep_last: number of frames to keep. Default: num_cond * 2.
        """
        if keep_last is None:
            keep_last = self.num_cond * 2
        if self.buffer is not None and self.buffer.shape[1] > keep_last:
            self.buffer = self.buffer[:, -keep_last:]

    def get_trajectory(self, dim: int = 0) -> list[float]:
        """Get mean value trajectory for a specific state dimension.

        Args:
            dim: state dimension index (0=position, 1=cash, ...).

        Returns:
            list of mean values per frame.
        """
        assert self.buffer is not None
        vals = self.buffer[0, :, :, :, dim]  # [T, H, W]
        if self.valid_mask is not None:
            vm = self.valid_mask.to(vals.device)
            return [vals[t][vm].mean().item() for t in range(vals.shape[0])]
        return [vals[t].mean().item() for t in range(vals.shape[0])]

    def rollout(self, n_rounds: int, trim_every: int = 5) -> torch.Tensor:
        """Run n_rounds of generation without intervention.

        Args:
            n_rounds: number of step() calls.
            trim_every: trim buffer every N rounds to save memory.

        Returns:
            all_generated: [n_rounds * num_gen, H, W, d_state]
        """
        all_gen = []
        for i in range(n_rounds):
            gen = self.step()
            all_gen.append(gen)
            if (i + 1) % trim_every == 0:
                self.trim_buffer()
        return torch.cat(all_gen, dim=0)
