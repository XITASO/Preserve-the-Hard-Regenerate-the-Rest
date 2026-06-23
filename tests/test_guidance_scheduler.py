import math
import torch

from guided_generation.guidance.guidance_scheduler import GuidanceScheduler


class _StubScheduler:
    def __init__(self, alphas_cumprod):
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self.config = type("Config", (), {"num_train_timesteps": len(alphas_cumprod)})()


def test_paper_early_schedule_uses_alpha_t():
    scheduler = GuidanceScheduler(
        base_eta=2.0,
        schedule_type="paper_early",
        diffusion_scheduler=_StubScheduler([1.0, 0.8, 0.5, 0.2]),
    )

    assert math.isclose(scheduler.get_eta(0), 2.0)
    assert math.isclose(scheduler.get_eta(1), 1.6, rel_tol=1e-6)
    assert math.isclose(scheduler.get_eta(2), 1.0)
    assert math.isclose(scheduler.get_eta(3), 0.4, rel_tol=1e-6)


def test_linear_schedule_uses_scheduler_train_timesteps():
    scheduler = GuidanceScheduler(
        base_eta=10.0,
        schedule_type="linear",
        diffusion_scheduler=_StubScheduler([1.0, 0.8, 0.5, 0.2]),
    )

    assert math.isclose(scheduler.get_eta(3), 10.0)
    assert math.isclose(scheduler.get_eta(0), 0.0)
