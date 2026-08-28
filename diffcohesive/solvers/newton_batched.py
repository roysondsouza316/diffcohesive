"""GPU-batched Newton solve over multiple specimens/test configurations simultaneously:
a batch of cohesive-law parameter sets (and/or initial history states)
sharing the same mesh/topology -- e.g. evaluating a population of candidate theta's during
calibration, or several mode-mix configurations at once.

Batching requires no per-element early-exit branching (``torch.func.vmap`` cannot trace
data-dependent control flow like ``if res_norm < tol: break``), so this runs a *fixed* number of
iterations for every batch element. Extra iterations on an already-converged element are cheap
near-no-ops (residual ~0 there, so the Newton update is ~0 too), not incorrect -- the
price of batching is giving up early-exit.
"""

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class BatchedNewtonResult:
    u: torch.Tensor          # (batch, n_dof)
    kappa: torch.Tensor      # (batch, n_coh, n_quad)
    damage: torch.Tensor     # (batch, n_coh, n_quad)
    reaction: torch.Tensor   # (batch, n_prescribed)


def _free_dofs(n_dof: int, prescribed_dofs: torch.Tensor, device) -> torch.Tensor:
    all_dofs = torch.arange(n_dof, device=device)
    is_fixed = torch.zeros(n_dof, dtype=torch.bool, device=device)
    is_fixed[prescribed_dofs] = True
    return all_dofs[~is_fixed]


def newton_solve_batched(
    model,
    prescribed_dofs: torch.Tensor,
    prescribed_values_batch: torch.Tensor,
    law_params_batch: Dict[str, torch.Tensor],
    kappa0_batch: torch.Tensor,
    max_iter: int = 30,
) -> BatchedNewtonResult:
    """
    Args:
        prescribed_values_batch: (batch, n_prescribed).
        law_params_batch: {name: (batch, *param_shape)} -- one theta per batch element,
            substituted functionally (no mutation of ``model.elem.law``'s live parameters).
        kappa0_batch: (batch, n_coh, n_quad) initial history, one per batch element.
    """
    dtype = model.points.dtype
    device = model.points.device
    n_dof = model.n_dof
    prescribed_dofs = prescribed_dofs.to(device)
    free_dofs = _free_dofs(n_dof, prescribed_dofs, device)

    def solve_one(prescribed_values, law_params, kappa0):
        # kappa0 (the history at the *start* of this load step) must stay fixed across every
        # Newton iteration within this step -- exactly like newton_solve's kappa_state, which
        # is never reassigned inside its iteration loop. Threading the just-computed kappa_new
        # back in every iteration (as an earlier version of this function did) advances the
        # damage history once per Newton iteration instead of once per load step, artificially
        # ratcheting damage up further with every extra iteration past convergence.
        u = torch.zeros(n_dof, dtype=dtype, device=device)
        u = u.index_copy(0, prescribed_dofs, prescribed_values)
        for _ in range(max_iter):
            R, _, _ = model.residual(u, kappa0, law_params=law_params)
            K = model.tangent(u.detach(), kappa0, law_params=law_params)
            K_ff = K[free_dofs.unsqueeze(-1), free_dofs.unsqueeze(0)]
            du_free = torch.linalg.solve(K_ff, -R[free_dofs].detach())
            u = u.index_copy(0, free_dofs, u[free_dofs] + du_free)
        R, kappa_new, damage = model.residual(u, kappa0, law_params=law_params)
        reaction = R[prescribed_dofs]
        return u, kappa_new, damage, reaction

    u, kappa, damage, reaction = torch.func.vmap(solve_one, in_dims=(0, 0, 0))(
        prescribed_values_batch, law_params_batch, kappa0_batch
    )
    return BatchedNewtonResult(u=u, kappa=kappa, damage=damage, reaction=reaction)
