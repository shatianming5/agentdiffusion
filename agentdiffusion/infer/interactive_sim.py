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
import torch.nn.functional as F
from ..models.video_dit import VideoDiT, VideoDDIMSampler
from ..diffusion.scheduler import NoiseScheduler


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
        recalibrate_every: int = 0,
        recalibrate_strength: float = 0.3,
        scheduler: NoiseScheduler | None = None,
        market_cond: torch.Tensor | None = None,
        anchor_state_stats: bool = False,
        target_state_mean: torch.Tensor | float | None = None,
        target_state_std: torch.Tensor | float | None = None,
    ):
        self.model = model
        self.sampler = sampler
        self.num_cond = num_cond
        self.num_gen = num_gen
        self.zero_sum_proj = zero_sum_proj
        self.valid_mask = valid_mask

        # SDEdit recalibration: add noise then denoise every N steps
        self.recalibrate_every = recalibrate_every
        self.recalibrate_strength = recalibrate_strength
        self.scheduler = scheduler

        self.device = next(model.parameters()).device
        if market_cond is not None and market_cond.dim() == 1:
            market_cond = market_cond.unsqueeze(0)
        self.market_cond = market_cond.to(self.device) if market_cond is not None else None
        self.anchor_state_stats = anchor_state_stats
        self.target_state_mean = self._to_device_scalar(target_state_mean)
        self.target_state_std = self._to_device_scalar(target_state_std)
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
        self._init_anchor_state_stats(self.buffer)
        self.total_steps = 0

    def _to_device_scalar(self, value: torch.Tensor | float | None) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().to(self.device)
        return torch.tensor(float(value), device=self.device)

    def _init_anchor_state_stats(self, frames: torch.Tensor) -> None:
        if not self.anchor_state_stats:
            return
        if self.target_state_mean is None:
            self.target_state_mean = frames.mean(dim=-1).mean().detach()
        if self.target_state_std is None:
            self.target_state_std = frames.std(dim=-1).mean().detach().clamp(min=1e-6)

    def _apply_zero_sum_projection(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.zero_sum_proj:
            return frames
        pos = frames[..., 0]
        if self.valid_mask is not None:
            vm = self.valid_mask.to(pos.device)
            n_valid = vm.sum().float().clamp(min=1)
            net = (pos * vm).sum(dim=(-2, -1), keepdim=True) / n_valid
            frames[..., 0] = pos - net * vm
        else:
            net = pos.mean(dim=(-2, -1), keepdim=True)
            frames[..., 0] = pos - net
        return frames

    def _anchor_generated_states(self, generated: torch.Tensor) -> torch.Tensor:
        if not self.anchor_state_stats:
            return generated
        assert self.target_state_mean is not None and self.target_state_std is not None
        # Keep autoregressive rollouts on the encoder's latent manifold by
        # matching the per-cell feature statistics of the initial seed states.
        anchored = F.layer_norm(generated, (generated.shape[-1],))
        anchored = anchored * self.target_state_std + self.target_state_mean
        return self._apply_zero_sum_projection(anchored)

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
                x_cond, gen_shape, market_cond=self.market_cond, device=self.device,
                zero_sum_proj=self.zero_sum_proj,
                valid_mask=self.valid_mask,
            )  # [1, N, H, W, D]
            generated = self._anchor_generated_states(generated)

        # Append to buffer (keep rolling)
        self.buffer = torch.cat([self.buffer, generated], dim=1)
        self.total_steps += 1

        # SDEdit recalibration: add noise → denoise to prevent mode collapse
        if (self.recalibrate_every > 0
                and self.total_steps % self.recalibrate_every == 0
                and self.scheduler is not None):
            self._recalibrate()

        return generated[0]  # [N, H, W, D]

    def _recalibrate(self):
        """SDEdit-style recalibration: add noise at strength t, then denoise."""
        K = self.num_cond
        if self.buffer.shape[1] <= K:
            return
        # Take the last few generated frames
        tail = self.buffer[:, -K:]  # [1, K, H, W, D]
        B, T, H, W, D = tail.shape
        # Add noise at a fraction of the full schedule
        t_recal = int(self.scheduler.timesteps * self.recalibrate_strength)
        t_tensor = torch.full((B * T,), t_recal, device=self.device, dtype=torch.long)
        flat = tail.reshape(B * T, H, W, D)
        noise = torch.randn_like(flat)
        noisy = self.scheduler.q_sample(flat, t_tensor, noise)
        # Denoise via single-step v-prediction
        with torch.no_grad():
            # Use the condition frames before tail
            cond_start = max(0, self.buffer.shape[1] - 2 * K)
            x_cond = self.buffer[:, cond_start:cond_start + K]
            gen_shape = (B, T, H, W, D)
            # Quick denoise: just run a few DDIM steps from the noisy state
            denoised = self.sampler.sample(
                x_cond, gen_shape, market_cond=self.market_cond, device=self.device,
                zero_sum_proj=self.zero_sum_proj,
                valid_mask=self.valid_mask,
            )
            denoised = self._anchor_generated_states(denoised)
        # Replace tail with denoised version
        self.buffer[:, -K:] = denoised[:, :K]

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
