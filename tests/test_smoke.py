"""Smoke tests: verify all modules can be imported and basic forward passes work."""

import math
import torch
import pytest


def test_agent_state():
    from agentdiffusion.data.agent_state import AgentGrid, AgentType, normalize_states, denormalize_states

    grid = AgentGrid(100)
    assert grid.grid_h * grid.grid_w >= 100

    states = torch.randn(100, 128)
    types = torch.randint(0, 4, (100,))
    capitals = torch.rand(100) * 1000

    grid_s, grid_t, sort_idx = grid.arrange(states, types, capitals)
    assert grid_s.shape == (grid.grid_h, grid.grid_w, 128)

    recovered = grid.flatten(grid_s, sort_idx)
    assert recovered.shape == (100, 128)

    normed, stats = normalize_states(states)
    denormed = denormalize_states(normed, stats)
    assert torch.allclose(states[:, 48:64], denormed[:, 48:64], atol=1e-5)


def test_autoencoder():
    from agentdiffusion.models.autoencoder import AgentAutoencoder

    ae = AgentAutoencoder(raw_dim=128, latent_dim=16)
    x = torch.randn(32, 128)
    out = ae(x)
    assert out["recon"].shape == (32, 128)
    assert out["z"].shape == (32, 16)
    assert out["recon_loss"].ndim == 0

    # VAE mode
    vae = AgentAutoencoder(raw_dim=128, latent_dim=16, vae=True)
    out = vae(x)
    assert "kl_loss" in out


def test_patchify():
    from agentdiffusion.models.patchify import PatchEmbedding, Unpatchify

    B, H, W, d_agent = 2, 16, 16, 16
    patch_size, d_model = 4, 64

    embed = PatchEmbedding(patch_size, d_agent, d_model, max_patches=256)
    x = torch.randn(B, H, W, d_agent)
    tokens = embed(x)
    assert tokens.shape == (B, (H // patch_size) * (W // patch_size), d_model)

    unpatch = Unpatchify(patch_size, d_agent, d_model)
    recovered = unpatch(tokens, H, W)
    assert recovered.shape == (B, H, W, d_agent)


def test_attention():
    from agentdiffusion.models.attention import MultiHeadAttention, CrossAttention, LocalWindowAttention

    B, N, D = 2, 64, 128
    mha = MultiHeadAttention(D, heads=4)
    out = mha(torch.randn(B, N, D))
    assert out.shape == (B, N, D)

    ca = CrossAttention(D, heads=4)
    out = ca(torch.randn(B, N, D), torch.randn(B, 8, D))
    assert out.shape == (B, N, D)

    lwa = LocalWindowAttention(D, heads=4, window_size=4)
    out = lwa(torch.randn(B, 64, D), 8, 8)
    assert out.shape == (B, 64, D)


def test_dit_block():
    from agentdiffusion.models.dit_block import DiTBlock

    B, N, D = 2, 16, 128
    block = DiTBlock(d_model=D, heads=4, num_market_tokens=8, local_window_size=4)
    x = torch.randn(B, N, D)
    c = torch.randn(B, D)
    out = block(x, c, Hp=4, Wp=4)
    assert out.shape == (B, N, D)


def test_agent_dit_forward():
    from agentdiffusion.models.agent_dit import AgentDiT

    B, H, W = 2, 16, 16
    model = AgentDiT(
        raw_dim=128, latent_dim=16, d_model=64, depth=2, heads=4,
        patch_size=4, num_market_tokens=8, local_window_size=4,
    )
    z = torch.randn(B, H, W, 16)
    t = torch.randint(0, 1000, (B,))
    mc = torch.randn(B, 32)
    out = model(z, t, mc)
    assert out.shape == (B, H, W, 16)


def test_scheduler():
    from agentdiffusion.diffusion.scheduler import NoiseScheduler

    sched = NoiseScheduler(timesteps=100, schedule="cosine")
    x0 = torch.randn(2, 8, 8, 16)
    t = torch.tensor([10, 50])
    x_t = sched.q_sample(x0, t)
    assert x_t.shape == x0.shape

    noise = torch.randn_like(x0)
    v = sched.v_target(x0, noise, t)
    x0_rec = sched.predict_x0_from_v(x_t, t, v)
    # x0_rec won't equal x0 exactly since x_t was sampled with different noise


def test_ddpm_loss():
    from agentdiffusion.models.agent_dit import AgentDiT
    from agentdiffusion.diffusion.scheduler import NoiseScheduler
    from agentdiffusion.diffusion.ddpm import DDPMTrainer

    model = AgentDiT(
        raw_dim=128, latent_dim=16, d_model=64, depth=2, heads=4,
        patch_size=4, num_market_tokens=8, local_window_size=4,
    )
    sched = NoiseScheduler(100, "cosine")
    trainer = DDPMTrainer(model, sched, "v_prediction")

    z0 = torch.randn(2, 16, 16, 16)
    out = trainer.compute_loss(z0)
    assert out["loss"].ndim == 0
    assert out["loss"].requires_grad


def test_ddim_sample():
    from agentdiffusion.models.agent_dit import AgentDiT
    from agentdiffusion.diffusion.scheduler import NoiseScheduler
    from agentdiffusion.diffusion.ddim import DDIMSampler

    model = AgentDiT(
        raw_dim=128, latent_dim=16, d_model=64, depth=2, heads=4,
        patch_size=4, num_market_tokens=8, local_window_size=4,
    )
    model.eval()
    sched = NoiseScheduler(100, "cosine")
    sampler = DDIMSampler(model, sched, "v_prediction", ddim_steps=5)
    sample = sampler.sample((1, 16, 16, 16))
    assert sample.shape == (1, 16, 16, 16)


def test_video_dit_condition_shapes():
    from agentdiffusion.models.video_dit import VideoDiT

    B, K, N = 2, 2, 3
    H, W, C = 8, 8, 16
    cond_dim = 12
    model = VideoDiT(
        d_latent=C,
        d_model=64,
        depth=2,
        heads=4,
        patch_size=2,
        grid_h=H,
        grid_w=W,
        num_frames=K + N,
        num_cond_frames=K,
        market_cond_dim=cond_dim,
    )

    x_cond = torch.randn(B, K, H, W, C)
    x_noisy = torch.randn(B, N, H, W, C)
    t = torch.randint(0, 1000, (B,))

    global_cond = torch.randn(B, cond_dim)
    out = model(x_cond, x_noisy, t, market_cond=global_cond)
    assert out.shape == (B, N, H, W, C)

    frame_cond = torch.randn(B, K + N, cond_dim)
    out = model(x_cond, x_noisy, t, market_cond=frame_cond)
    assert out.shape == (B, N, H, W, C)

    grid_cond = torch.randn(B, K + N, H, W, cond_dim)
    out = model(x_cond, x_noisy, t, market_cond=grid_cond)
    assert out.shape == (B, N, H, W, C)


def test_constraint_loss():
    from agentdiffusion.constraints.soft_loss import ConstraintLoss

    cl = ConstraintLoss()
    state_t = torch.randn(2, 8, 8, 128)
    state_t1 = torch.randn(2, 8, 8, 128)
    out = cl(state_t, state_t1)
    assert "constraint_total" in out
    assert out["constraint_total"].ndim == 0


def test_projection():
    from agentdiffusion.constraints.projection import apply_all_projections

    state_t = torch.randn(2, 8, 8, 128)
    state_t1 = torch.randn(2, 8, 8, 128)
    projected = apply_all_projections(state_t, state_t1)
    assert projected.shape == state_t1.shape

    # Market clearing: net delta should be ~zero
    delta = projected[..., :32] - state_t[..., :32]
    net = delta.sum(dim=(1, 2))
    assert net.abs().max() < 1e-5


def test_stylized_facts():
    from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts

    np_gen = __import__("numpy").random.default_rng(42)
    prices = 100 + np_gen.standard_normal(1000).cumsum() * 0.5
    prices = prices.clip(min=1)

    report = evaluate_stylized_facts(prices)
    assert hasattr(report, "total_passed")
    assert 0 <= report.total_passed <= 6


def test_synthetic_dataset():
    from agentdiffusion.data.dataset import SyntheticAgentDataset

    ds = SyntheticAgentDataset(num_samples=10, grid_h=8, grid_w=8)
    sample = ds[0]
    assert sample["state_t"].shape == (8, 8, 128)
    assert sample["state_t1"].shape == (8, 8, 128)
    assert sample["market_cond"].shape == (32,)


def test_config():
    from agentdiffusion.utils.config import load_config

    cfg = load_config()
    assert cfg.agent.raw_dim == 128
    assert cfg.model.d_model == 512
