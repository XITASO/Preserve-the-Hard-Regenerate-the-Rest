from typing import Any

import torch


class GuidanceScheduler:
    """
    Timestep progresses from high-noise to clean.
    Selects a guidance factor dynamically based on the configured schedule.
    """

    def __init__(
        self,
        base_eta: float,
        schedule_type: str = "constant",
        diffusion_scheduler: Any = None,
    ):
        self.base_eta = base_eta
        self.diffusion_scheduler = diffusion_scheduler
        self.scheduler_function = self._set_scheduler_function(schedule_type=schedule_type)

    def _set_scheduler_function(self, schedule_type: str):
        """By default, every scheduler function is designed such that its maximum value is 1."""
        available_schedulers = {
            "constant": self._const_schedule,
            "linear": self._linear_schedule,
            # Paper-style early schedule: gamma_early(t) = alpha_t.
            "paper_early": self._paper_early_schedule,
        }
        if schedule_type in available_schedulers:
            return available_schedulers[schedule_type]

        raise ValueError(
            f"Invalid schedule_type '{schedule_type}' provided. Available types are: {', '.join(available_schedulers.keys())}."
        )

    def _to_int_timestep(self, timestep: int) -> int:
        if isinstance(timestep, torch.Tensor):
            timestep = int(timestep.detach().cpu().item())
        return int(timestep)

    def _resolve_num_train_timesteps(self) -> int:
        if self.diffusion_scheduler is None:
            return 1000

        scheduler_config = getattr(self.diffusion_scheduler, "config", None)
        if scheduler_config is None:
            return 1000

        return int(getattr(scheduler_config, "num_train_timesteps", 1000))

    def _linear_schedule(self, timestep: int) -> float:
        t = self._to_int_timestep(timestep)
        num_train_timesteps = self._resolve_num_train_timesteps()
        t = max(0, min(t, num_train_timesteps - 1))
        return t / max(1, num_train_timesteps - 1)

    def _const_schedule(self, timestep: int) -> float:
        return 1.0

    def _paper_early_schedule(self, timestep: int) -> float:
        if self.diffusion_scheduler is None or not hasattr(self.diffusion_scheduler, "alphas_cumprod"):
            raise ValueError(
                "Schedule 'paper_early' requires a diffusion scheduler with `alphas_cumprod`."
            )

        t = self._to_int_timestep(timestep)
        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod
        t = max(0, min(t, len(alphas_cumprod) - 1))

        alpha_t = alphas_cumprod[t]
        if isinstance(alpha_t, torch.Tensor):
            alpha_t = float(alpha_t.detach().cpu().item())
        else:
            alpha_t = float(alpha_t)

        alpha_t = min(max(alpha_t, 0.0), 1.0)

        # Paper early scheduler:
        # gamma_early(t) = alpha_t
        return alpha_t

    def get_eta(self, timestep: int) -> float:
        return self.base_eta * self.scheduler_function(timestep)
