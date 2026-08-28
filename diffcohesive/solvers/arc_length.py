"""Crisfield/Riks arc-length path-following. Needed
because force-controlled Newton cannot pass a snap-back peak: past peak load, no equilibrium
solution exists for a further *increase* in the prescribed load, only for a decrease. Arc-length
adds an extra unknown (the load factor lambda) and a constraint on the combined step length, so
the solver can trace load *and* displacement through the snap-back.

Cylindrical variant (load-term weight psi = 0): the constraint is on ||Delta u||^2 = ds^2 alone.
All DOFs in ``fixed_dofs`` are held at zero displacement throughout (homogeneous Dirichlet);
``f_hat`` is the fixed load *pattern*, scaled at each step by the solved-for load factor lambda.
"""

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ArcLengthStep:
    u: torch.Tensor
    lam: float
    kappa: torch.Tensor
    damage: torch.Tensor
    converged: bool
    n_iter: int


def arc_length_solve(
    model,
    fixed_dofs: torch.Tensor,
    f_hat: torch.Tensor,
    kappa_state: torch.Tensor,
    ds: float,
    n_steps: int,
    tol: float = 1e-9,
    max_iter: int = 40,
    u0: torch.Tensor = None,
    lam0: float = 0.0,
    adaptive: bool = True,
    max_cuts: int = 8,
) -> List[ArcLengthStep]:
    """``u0``/``lam0`` optionally warm-start from a converged state (e.g. handing off from a
    displacement-controlled ``newton_solve`` run once its prescribed DOFs are released to become
    part of ``free`` here, at load level ``lam0`` along the same ``f_hat`` pattern), instead of
    always starting from the unloaded (u=0, lambda=0) state.

    ``adaptive``: on a non-converged step, retry the same step with the arc increment halved
    (up to ``max_cuts`` times, standard Crisfield practice -- needed to trace the small local
    snap-backs of successive cohesive-element failures), growing back gently after successes.
    The analysis only terminates early if the increment is cut ``max_cuts`` times in a row.
    """
    dtype = model.points.dtype
    device = model.points.device
    n_dof = model.n_dof
    fixed_dofs = fixed_dofs.to(device)
    f_hat = f_hat.to(device)
    all_dofs = torch.arange(n_dof, device=device)
    is_fixed = torch.zeros(n_dof, dtype=torch.bool, device=device)
    is_fixed[fixed_dofs] = True
    free = all_dofs[~is_fixed]

    u = torch.zeros(n_dof, dtype=dtype, device=device) if u0 is None else u0.clone()
    lam = lam0
    kappa = kappa_state.clone()
    f_free = f_hat[free]

    history: List[ArcLengthStep] = []
    prev_du_free = None
    ds_current = ds
    n_cuts = 0

    step = 0
    while step < n_steps:
        K = model.tangent(u.detach(), kappa)
        K_ff = K[free.unsqueeze(-1), free.unsqueeze(0)]
        if not torch.isfinite(K_ff).all():
            break  # converged state produced a non-finite tangent: nothing sane to continue from
        du_bar = torch.linalg.solve(K_ff, f_free)

        sign = 1.0
        if prev_du_free is not None:
            sign = 1.0 if torch.dot(du_bar, prev_du_free).item() >= 0 else -1.0
        dlam = sign * ds_current / du_bar.norm().item()
        du_free = dlam * du_bar

        u_iter = u.clone()
        u_iter[free] = u_iter[free] + du_free
        lam_iter = lam + dlam
        du_accum = du_free.clone()
        dlam_accum = dlam

        converged = False
        n_iter = 0
        res_norm_first = None
        for n_iter in range(1, max_iter + 1):
            # Guard the trial state itself: an extreme (even merely huge, not yet inf) u_iter
            # can crash the native vmap/jacrev/LAPACK kernels on Windows rather than raise.
            # Treat it as a failed step and let the adaptive increment cut take over.
            if not torch.isfinite(u_iter).all() or u_iter.abs().max().item() > 1.0e10:
                break
            R, kappa_new, damage = model.residual(u_iter, kappa)
            res = R[free] - lam_iter * f_free
            res_norm = res.norm().item()
            # Force-scale-relative convergence (see newton.py): absolute tol alone falls
            # below the numerical residual floor of larger models.
            scale = max(1.0, abs(lam_iter) * f_free.norm().item())
            if res_norm < tol * scale:
                converged = True
                break
            # Early divergence detection: a step that is going to fail spends all max_iter
            # iterations discovering it, which multiplied by the adaptive increment cuts makes
            # failed steps dominate wall-clock. Abort as soon as the residual has clearly blown
            # up relative to the first corrector iteration (or went non-finite).
            if res_norm != res_norm or res_norm == float("inf"):  # NaN/inf
                break
            if res_norm_first is None:
                res_norm_first = res_norm
            elif res_norm > 1.0e6 * max(res_norm_first, tol):
                break  # violent blow-up: abort immediately
            elif n_iter > 5 and res_norm > 1.0e3 * max(res_norm_first, tol):
                break  # steady divergence: give up after a few corrector attempts

            try:
                K = model.tangent(u_iter.detach(), kappa)
            except FloatingPointError:
                break  # non-finite state inside a failed increment: cut and retry
            K_ff = K[free.unsqueeze(-1), free.unsqueeze(0)]
            # Non-finite tangent entries (an extreme trial state deep in a failed increment)
            # hard-crash MKL's LAPACK on Windows (access violation, not a Python exception) --
            # bail out and let the adaptive increment cut handle it instead.
            if not torch.isfinite(K_ff).all():
                break
            rhs = torch.stack([-res.detach(), f_free], dim=-1)
            sol = torch.linalg.solve(K_ff, rhs)  # one factorization for both corrector solves
            du_r = sol[:, 0]
            du_f = sol[:, 1]

            base = du_accum + du_r
            a = torch.dot(du_f, du_f).item()
            b = 2.0 * torch.dot(base, du_f).item()
            c = torch.dot(base, base).item() - ds_current * ds_current
            disc = max(b * b - 4 * a * c, 0.0)
            sqrt_disc = disc ** 0.5
            root1 = (-b + sqrt_disc) / (2 * a)
            root2 = (-b - sqrt_disc) / (2 * a)

            cand1 = base + root1 * du_f
            cand2 = base + root2 * du_f
            dot1 = torch.dot(cand1, du_accum).item()
            dot2 = torch.dot(cand2, du_accum).item()
            droot = root1 if dot1 >= dot2 else root2
            dcand = cand1 if dot1 >= dot2 else cand2

            du_accum = dcand
            dlam_accum += droot
            u_iter = u.clone()
            u_iter[free] = u_iter[free] + du_accum
            lam_iter = lam + dlam_accum

        if not converged and adaptive:
            # Increment cut: retry the SAME step from the same converged state with half the
            # arc increment (u/lam/kappa/prev_du_free untouched), unless already cut to dust.
            n_cuts += 1
            if n_cuts <= max_cuts:
                ds_current *= 0.5
                continue

        R, kappa_new, damage = model.residual(u_iter, kappa)
        u = u_iter
        lam = lam_iter
        kappa = kappa_new
        prev_du_free = du_accum
        history.append(ArcLengthStep(u=u.clone(), lam=lam, kappa=kappa.clone(), damage=damage.clone(), converged=converged, n_iter=n_iter))
        if not converged:
            break
        step += 1
        n_cuts = 0
        if adaptive:
            ds_current = min(ds_current * 1.2, ds)

    return history
