"""Gradient-based calibration loop: Adam to get into the basin, then L-BFGS to
polish, both driven by the implicit-diff adjoint (diffcohesive.diff). Calibrated components are
optimized in log-space so they stay positive (T_max, G_c, K are all physically positive); any
components not in ``free_mask`` are held fixed at their initial value.
"""

from typing import Callable, Tuple

import torch


def calibrate(
    theta_init: torch.Tensor,
    free_mask: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    n_adam: int = 100,
    lr_adam: float = 0.05,
    n_lbfgs: int = 30,
    history: list = None,
) -> Tuple[torch.Tensor, float]:
    """``history``: optional list; when given, the loss value of every optimizer evaluation
    (each of which costs exactly one forward solve + one adjoint solve) is appended, so
    convergence can be plotted against the forward-solve count on the same axis as the GA
    baseline's evaluation count."""
    phi = torch.log(theta_init[free_mask]).clone().requires_grad_(True)

    def build_theta(phi_val: torch.Tensor) -> torch.Tensor:
        return theta_init.clone().masked_scatter(free_mask, torch.exp(phi_val))

    adam = torch.optim.Adam([phi], lr=lr_adam)
    for _ in range(n_adam):
        adam.zero_grad()
        loss = loss_fn(build_theta(phi))
        loss.backward()
        adam.step()
        if history is not None:
            history.append(loss.item())

    lbfgs = torch.optim.LBFGS([phi], max_iter=n_lbfgs, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = loss_fn(build_theta(phi))
        loss.backward()
        if history is not None:
            history.append(loss.item())
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        final_loss = loss_fn(build_theta(phi)).item()
    return build_theta(phi).detach(), final_loss
