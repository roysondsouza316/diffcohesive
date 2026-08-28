"""Derivative-free genetic-algorithm calibration baseline, on the same forward model as the
gradient-based calibrator (for cost/robustness comparison). Free
parameters are searched in log-space (matching diffcohesive.calibration.calibrate's
parametrization) so both methods explore the same positivity-constrained space.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import torch


@dataclass
class GAResult:
    theta: torch.Tensor
    loss: float
    n_evals: int
    loss_history: List[float] = field(default_factory=list)


def genetic_algorithm(
    theta_init: torch.Tensor,
    free_mask: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    log_bounds: Tuple[float, float] = (-2.0, 2.0),
    pop_size: int = 24,
    n_generations: int = 60,
    elite_frac: float = 0.15,
    mutation_std: float = 0.3,
    seed: int = 0,
) -> GAResult:
    """log_bounds are relative to log(theta_init[free_mask]) (i.e. search within
    theta_init * exp([-2,2]) by default, a ~e^4 ~ 55x range around the initial guess)."""
    gen = torch.Generator().manual_seed(seed)
    n_free = int(free_mask.sum().item())
    log_theta0_free = torch.log(theta_init[free_mask])

    def build_theta(log_free: torch.Tensor) -> torch.Tensor:
        return theta_init.clone().masked_scatter(free_mask, torch.exp(log_free))

    def evaluate(log_free: torch.Tensor) -> float:
        try:
            with torch.no_grad():
                return loss_fn(build_theta(log_free)).item()
        except RuntimeError:
            return 1.0e6  # non-convergent forward solve at this (degenerate) theta: penalize

    low, high = log_bounds
    population = log_theta0_free.unsqueeze(0) + (
        torch.rand(pop_size, n_free, generator=gen, dtype=theta_init.dtype) * (high - low) + low
    )
    n_evals = 0
    losses = torch.tensor([evaluate(population[i]) for i in range(pop_size)], dtype=theta_init.dtype)
    n_evals += pop_size
    history = [losses.min().item()]

    n_elite = max(1, int(elite_frac * pop_size))
    for _ in range(n_generations):
        order = torch.argsort(losses)
        population = population[order]
        losses = losses[order]

        new_pop = [population[:n_elite]]
        while sum(p.shape[0] for p in new_pop) < pop_size:
            i, j = torch.randint(0, n_elite * 2 if n_elite * 2 < pop_size else pop_size, (2,), generator=gen)
            i, j = i.item(), j.item()
            alpha = torch.rand((), generator=gen, dtype=theta_init.dtype)
            child = alpha * population[i] + (1 - alpha) * population[j]
            child = child + mutation_std * torch.randn(n_free, generator=gen, dtype=theta_init.dtype)
            child = child.clamp(log_theta0_free.min().item() + low, log_theta0_free.max().item() + high)
            new_pop.append(child.unsqueeze(0))
        population = torch.cat(new_pop, dim=0)[:pop_size]

        losses = torch.tensor([evaluate(population[i]) for i in range(pop_size)], dtype=theta_init.dtype)
        n_evals += pop_size
        history.append(losses.min().item())

    best_idx = torch.argmin(losses)
    return GAResult(theta=build_theta(population[best_idx]), loss=losses[best_idx].item(), n_evals=n_evals, loss_history=history)
