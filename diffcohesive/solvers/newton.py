"""Newton-Raphson solver for the pre-peak (non-softening) regime. Displacement-
controlled: a set of DOFs carry prescribed (possibly load-factor-scaled) values; the rest are
free and solved for equilibrium with zero external force on them.
"""

from dataclasses import dataclass

import torch


@dataclass
class NewtonResult:
    u: torch.Tensor
    kappa: torch.Tensor
    damage: torch.Tensor
    converged: bool
    n_iter: int
    reaction: torch.Tensor  # residual at the prescribed dofs (reaction force)


def newton_solve(
    model,
    prescribed_dofs: torch.Tensor,
    prescribed_values: torch.Tensor,
    kappa_state: torch.Tensor,
    u0: torch.Tensor = None,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> NewtonResult:
    dtype = model.points.dtype
    device = model.points.device
    n_dof = model.n_dof
    prescribed_dofs = prescribed_dofs.to(device)
    prescribed_values = prescribed_values.to(device)
    all_dofs = torch.arange(n_dof, device=device)
    is_prescribed = torch.zeros(n_dof, dtype=torch.bool, device=device)
    is_prescribed[prescribed_dofs] = True
    free_dofs = all_dofs[~is_prescribed]

    u = torch.zeros(n_dof, dtype=dtype, device=device) if u0 is None else u0.clone()
    u[prescribed_dofs] = prescribed_values

    converged = False
    n_iter = 0
    R = None
    for n_iter in range(1, max_iter + 1):
        R, kappa_new, damage = model.residual(u, kappa_state)
        R_free = R[free_dofs]
        res_norm = R_free.norm().item()
        if res_norm < tol:
            converged = True
            break
        K = model.tangent(u.detach(), kappa_state)
        K_ff = K[free_dofs.unsqueeze(-1), free_dofs.unsqueeze(0)]
        du_free = torch.linalg.solve(K_ff, -R_free.detach())
        u = u.clone()
        u[free_dofs] = u[free_dofs] + du_free

    R, kappa_new, damage = model.residual(u, kappa_state)
    reaction = R[prescribed_dofs]
    return NewtonResult(u=u, kappa=kappa_new, damage=damage, converged=converged, n_iter=n_iter, reaction=reaction)
