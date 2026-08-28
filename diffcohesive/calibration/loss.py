"""Calibration loss and forward-response helper: L(theta) = ||P_sim(theta) - P_exp||^2
over a load-displacement curve, with P_sim obtained through the implicit-diff adjoint solve so
gradients flow via backprop instead of a derivative-free search.
"""

from typing import Sequence

import torch

from ..diff.implicit_diff import solve_diff, solve_diff_path, split_theta


def simulated_response(
    theta: torch.Tensor,
    model,
    prescribed_dofs: torch.Tensor,
    reaction_dof: torch.Tensor,
    disp_values: Sequence[float],
    fixed_prefix: torch.Tensor,
) -> torch.Tensor:
    """P_sim(d; theta) for each displacement in ``disp_values``, one independent single-step
    equilibrium solve per point (scope note: not chained across load history; see calibration_loss_path for that)."""
    kappa0 = model.init_history()
    u0 = torch.zeros(model.n_dof, dtype=theta.dtype, device=theta.device)
    responses = []
    for d in disp_values:
        prescribed_values = torch.cat([fixed_prefix, torch.tensor([d], dtype=theta.dtype, device=theta.device)])
        u_star = solve_diff(theta, model, prescribed_dofs, prescribed_values, kappa0, u0=u0)
        law_params = split_theta(model.elem.law, theta)
        R, _, _ = model.residual(u_star, kappa0, law_params=law_params)
        responses.append(R[reaction_dof])
    return torch.stack(responses)


def calibration_loss(
    theta: torch.Tensor,
    model,
    prescribed_dofs: torch.Tensor,
    reaction_dof: torch.Tensor,
    disp_values: Sequence[float],
    fixed_prefix: torch.Tensor,
    P_exp: torch.Tensor,
) -> torch.Tensor:
    P_sim = simulated_response(theta, model, prescribed_dofs, reaction_dof, disp_values, fixed_prefix)
    return ((P_sim - P_exp) ** 2).mean()


def simulated_response_path(
    theta: torch.Tensor,
    model,
    prescribed_dofs: torch.Tensor,
    reaction_dof: torch.Tensor,
    disp_values: Sequence[float],
    fixed_prefix: torch.Tensor,
) -> torch.Tensor:
    """P_sim(d; theta) for *one continuous* loading path (the multi-step scope extension of
    ``simulated_response``, via ``solve_diff_path``): history correctly carries over between
    displacement steps instead of resetting to kappa0 at every point independently, so this
    matches calibrating against one real continuous experimental curve rather than independent
    single-step equilibrium points."""
    kappa0 = model.init_history()
    u0 = torch.zeros(model.n_dof, dtype=theta.dtype, device=theta.device)
    dofs_seq = [prescribed_dofs for _ in disp_values]
    values_seq = [torch.cat([fixed_prefix, torch.tensor([d], dtype=theta.dtype, device=theta.device)]) for d in disp_values]
    u_list, kappa_list = solve_diff_path(theta, model, dofs_seq, values_seq, kappa0, u0=u0)

    law_params = split_theta(model.elem.law, theta)
    responses = []
    kappa_prev = kappa0
    for u, kappa in zip(u_list, kappa_list):
        R, _, _ = model.residual(u, kappa_prev, law_params=law_params)
        responses.append(R[reaction_dof])
        kappa_prev = kappa
    return torch.stack(responses)


def calibration_loss_path(
    theta: torch.Tensor,
    model,
    prescribed_dofs: torch.Tensor,
    reaction_dof: torch.Tensor,
    disp_values: Sequence[float],
    fixed_prefix: torch.Tensor,
    P_exp: torch.Tensor,
) -> torch.Tensor:
    P_sim = simulated_response_path(theta, model, prescribed_dofs, reaction_dof, disp_values, fixed_prefix)
    return ((P_sim - P_exp) ** 2).mean()
