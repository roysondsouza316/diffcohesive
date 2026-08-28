"""Implicit differentiation through the converged nonlinear solve, as a custom
``torch.autograd.Function``: forward runs Newton with gradient tracking off; backward uses the
implicit function theorem at the converged state,

    (dR/du)^T mu = dL/du            (one adjoint linear solve, reusing the converged tangent K)
    dL/dtheta = -mu^T (dR/dtheta)   (a vector-Jacobian product via autograd)

instead of backpropagating through the Newton iterations themselves (which would mean memory blowup
and noisy gradients). ``ImplicitNewtonSolve.backward`` returns a gradient for both ``theta`` and
``kappa_state`` (the VJP mu^T(dR/dtheta) and mu^T(dR/d(kappa_state)) respectively), so
``solve_diff_path`` below can chain steps where each step's initial history is itself a
theta-dependent output of the previous step's solve -- not just a fixed constant.
"""

from typing import List, Tuple

import torch

from ..solvers.newton import newton_solve


def theta_from_law(law: torch.nn.Module) -> torch.Tensor:
    """Flatten a law's parameters into a single theta vector (order = named_parameters())."""
    return torch.cat([p.detach().reshape(-1) for _, p in law.named_parameters()])


def _param_shapes(law: torch.nn.Module) -> List[Tuple[str, torch.Size, int]]:
    return [(name, p.shape, p.numel()) for name, p in law.named_parameters()]


def split_theta(law: torch.nn.Module, theta: torch.Tensor):
    """Public: map a flat theta vector to a {name: tensor} dict shaped like law's own parameters,
    for use with torch.func.functional_call (e.g. to compose a loss with both a direct and an
    implicit-via-u* dependence on theta)."""
    params = {}
    idx = 0
    for name, shape, numel in _param_shapes(law):
        params[name] = theta[idx : idx + numel].reshape(shape)
        idx += numel
    return params


def assign_theta_(law: torch.nn.Module, theta: torch.Tensor) -> None:
    """Public: in-place copy theta's values into law's own parameters (no grad tracking)."""
    named = dict(law.named_parameters())
    idx = 0
    with torch.no_grad():
        for name, shape, numel in _param_shapes(law):
            named[name].copy_(theta[idx : idx + numel].reshape(shape))
            idx += numel


def _free_dofs(n_dof: int, prescribed_dofs: torch.Tensor) -> torch.Tensor:
    device = prescribed_dofs.device
    all_dofs = torch.arange(n_dof, device=device)
    is_fixed = torch.zeros(n_dof, dtype=torch.bool, device=device)
    is_fixed[prescribed_dofs] = True
    return all_dofs[~is_fixed]


class ImplicitNewtonSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, model, prescribed_dofs, prescribed_values, kappa_state, u0):
        law = model.elem.law
        with torch.no_grad():
            assign_theta_(law, theta)
            result = newton_solve(model, prescribed_dofs, prescribed_values, kappa_state, u0=u0)
            if not result.converged:
                raise RuntimeError("ImplicitNewtonSolve: forward Newton solve did not converge")

        ctx.save_for_backward(result.u.detach(), theta.detach())
        ctx.model = model
        ctx.prescribed_dofs = prescribed_dofs
        ctx.kappa_state = kappa_state.detach()
        ctx.kappa_new = result.kappa.detach()
        ctx.damage = result.damage.detach()
        return result.u.detach()

    @staticmethod
    def backward(ctx, grad_u):
        u_star, theta_used = ctx.saved_tensors
        model = ctx.model
        kappa_state = ctx.kappa_state
        free = _free_dofs(model.n_dof, ctx.prescribed_dofs)

        K = model.tangent(u_star, kappa_state)
        K_ff = K[free.unsqueeze(-1), free.unsqueeze(0)]
        mu = torch.linalg.solve(K_ff.transpose(0, 1), grad_u[free])

        law = model.elem.law
        with torch.enable_grad():
            # Function.backward runs under no_grad by default; the VJP for dR/dtheta (and, for
            # multi-step chaining, dR/d(kappa_state)) needs a fresh autograd graph w.r.t. leaves
            # disconnected from the outer computation.
            theta_leaf = theta_used.clone().requires_grad_(True)
            kappa_leaf = kappa_state.clone().requires_grad_(True)
            law_params = split_theta(law, theta_leaf)
            R, _, _ = model.residual(u_star, kappa_leaf, law_params=law_params)
            grad_theta, grad_kappa = torch.autograd.grad(
                R[free], (theta_leaf, kappa_leaf), grad_outputs=mu
            )

        # grad_kappa lets solve_diff_path chain the adjoint across load steps: kappa_state's own
        # theta-dependence (from an earlier step's damage evolution) now receives a gradient
        # here, instead of being silently dropped as a constant.
        return -grad_theta, None, None, None, -grad_kappa, None


def solve_diff(
    theta: torch.Tensor,
    model,
    prescribed_dofs: torch.Tensor,
    prescribed_values: torch.Tensor,
    kappa_state: torch.Tensor,
    u0: torch.Tensor = None,
) -> torch.Tensor:
    """Differentiable wrapper: dL/dtheta flows through the adjoint, not through Newton iterates."""
    if u0 is None:
        u0 = torch.zeros(model.n_dof, dtype=theta.dtype, device=theta.device)
    return ImplicitNewtonSolve.apply(theta, model, prescribed_dofs, prescribed_values, kappa_state, u0)


def solve_diff_path(
    theta: torch.Tensor,
    model,
    prescribed_dofs_seq: List[torch.Tensor],
    prescribed_values_seq: List[torch.Tensor],
    kappa0: torch.Tensor,
    u0: torch.Tensor = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Chain ``solve_diff`` across a sequence of load steps. Each step's history state is
    computed from the *previous* step's (differentiable) displacement via the law itself, not
    detached, so gradients from a later step's loss correctly flow back through every earlier
    step's contribution to the current damage state -- the multi-step scope extension of
    ``solve_diff`` (single fixed-history step only).

    Returns ``(u_list, kappa_list)``, one entry per step, both differentiable w.r.t. ``theta``.
    """
    if u0 is None:
        u0 = torch.zeros(model.n_dof, dtype=theta.dtype, device=theta.device)
    law = model.elem.law

    u = u0
    kappa = kappa0
    u_list, kappa_list = [], []
    for prescribed_dofs, prescribed_values in zip(prescribed_dofs_seq, prescribed_values_seq):
        u = solve_diff(theta, model, prescribed_dofs, prescribed_values, kappa, u0=u.detach())
        law_params = split_theta(law, theta)
        _, kappa, _ = model.residual(u, kappa, law_params=law_params)
        u_list.append(u)
        kappa_list.append(kappa)
    return u_list, kappa_list
