"""Adaptive displacement-controlled stepping with automatic increment cutting -- the same
strategy the CZM validation literature uses for exactly this failure mode (e.g. Alfano CST
2006 §3: "When the analysis failed to converge in an increment after 12 iterations, it was
restarted with a halved increment size (increment cut)"): near the peak, individual cohesive
elements fail in small local snap-throughs that a fixed-size displacement increment steps
over, making Newton diverge even though the global displacement-controlled response is stable.
Halving the increment walks through the pop-in; the increment grows back afterwards.
"""

from typing import Callable, Optional, Tuple

import torch

from .newton import NewtonResult, newton_solve


def adaptive_displacement_solve(
    model,
    prescribed_fn: Callable[[float], Tuple[torch.Tensor, torch.Tensor]],
    kappa_state: torch.Tensor,
    u0: torch.Tensor,
    d_from: float,
    d_to: float,
    initial_step: Optional[float] = None,
    max_halvings: int = 10,
    growth: float = 1.5,
    **newton_kwargs,
) -> Optional[Tuple[NewtonResult, torch.Tensor, torch.Tensor, float]]:
    """Advance the load parameter from ``d_from`` to ``d_to`` with increment cutting.

    ``prescribed_fn(d)`` returns (prescribed_dofs, prescribed_values) for load level d.
    Returns (last_result, u, kappa, d_reached); d_reached < d_to only if the increment was
    cut ``max_halvings`` times without convergence (analysis stuck). Returns None if not even
    one increment converged.
    """
    span = d_to - d_from
    step = initial_step if initial_step is not None else span
    min_step = span / (2 ** max_halvings)

    d = d_from
    u = u0.clone()
    kappa = kappa_state.clone()
    last = None
    while d < d_to - 1e-12 * max(abs(d_to), 1.0):
        trial = min(d + step, d_to)
        pd, pv = prescribed_fn(trial)
        result = newton_solve(model, pd, pv, kappa, u0=u, **newton_kwargs)
        if result.converged:
            d = trial
            u, kappa = result.u, result.kappa
            last = result
            step = min(step * growth, d_to - d) if d < d_to else step
        else:
            step *= 0.5
            if step < min_step:
                break
    if last is None:
        return None
    return last, u, kappa, d
