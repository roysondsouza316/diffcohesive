"""Nonlinear bulk material combined with cohesive elements, through the
``bulk_residual_fn``/``bulk_tangent_fn`` hooks of ``CohesiveMeshModel``.

The arms of a 2D DCB are given a Ramberg-Osgood type nonlinear stress-strain response
(deformation theory: a secant modulus that decreases with an equivalent strain measure), a
simple stand-in for elastic-plastic arm behavior under monotonic loading. The bulk residual
and its consistent tangent are assembled per constant-strain triangle with the analytic
per-element material tangent, so the demonstration stays fast. The same damaging solve and
the same adjoint gradient machinery run unchanged: the solvers and the implicit
differentiation only ever see the total residual and tangent.

Outputs:
  examples/nonlinear_bulk_dcb.png   elastic vs nonlinear-arm load-displacement curves
  console                           adjoint gradient dP/dG_c1 vs finite differences, with
                                    the NONLINEAR bulk active

Run from repo root: PYTHONPATH=. python examples/nonlinear_bulk_dcb.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.diff import solve_diff, theta_from_law
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import build_double_cantilever_mesh
from diffcohesive.solvers import newton_solve

# geometry/material: quickstart-sized DCB
LENGTH, ARM, A0, NX, NY = 15.0, 1.0, 5.0, 30, 4
E, NU = 1000.0, 0.3
SIGMA0, GC, K0 = 5.0, 0.05, 1.0e4
# Saturating-hardening secant: E_sec(eps_eq) = E / sqrt(1 + eps_eq/eps_y), a smooth
# monotone stiffness reduction beyond the reference strain eps_y (the stress remains a
# monotonically increasing function of strain, so the bulk tangent stays positive
# definite; a monotonic-loading stand-in for hardening plasticity).
EPS_Y = 2.0e-3


def _cst_operators(points, tris):
    """Per-element area and B-matrix (constant-strain triangles, Voigt [xx, yy, xy])."""
    p = points[tris]                                       # (n_el, 3, 2)
    x, y = p[..., 0], p[..., 1]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    area = 0.5 * det.abs()
    b = torch.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], dim=1) / det.unsqueeze(1)
    c = torch.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], dim=1) / det.unsqueeze(1)
    n_el = tris.shape[0]
    B = torch.zeros(n_el, 3, 6, dtype=points.dtype)
    for a in range(3):
        B[:, 0, 2 * a] = b[:, a]
        B[:, 1, 2 * a + 1] = c[:, a]
        B[:, 2, 2 * a] = c[:, a]
        B[:, 2, 2 * a + 1] = b[:, a]
    return area, B


def make_nonlinear_bulk(points, tris, n_dof):
    """Saturating-hardening plane-strain bulk: residual R(u) and consistent tangent K(u)."""
    area, B = _cst_operators(points, tris)
    dofs = (tris.unsqueeze(-1) * 2 + torch.arange(2)).reshape(tris.shape[0], 6)
    lam0 = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu0 = E / (2 * (1 + NU))
    C0 = torch.tensor([[lam0 + 2 * mu0, lam0, 0.0],
                       [lam0, lam0 + 2 * mu0, 0.0],
                       [0.0, 0.0, mu0]], dtype=points.dtype)

    def strain(u):
        u_e = u[dofs]                                       # (n_el, 6)
        return torch.einsum("eij,ej->ei", B, u_e)           # (n_el, 3)

    def secant_scale(eps):
        eps_eq = torch.sqrt(eps[:, 0] ** 2 + eps[:, 1] ** 2 + 0.5 * eps[:, 2] ** 2 + 1e-24)
        return 1.0 / torch.sqrt(1.0 + eps_eq / EPS_Y)

    def residual(u):
        eps = strain(u)
        sig = secant_scale(eps).unsqueeze(1) * torch.einsum("ij,ej->ei", C0, eps)
        f_e = torch.einsum("e,eij,ei->ej", area, B, sig)
        R = torch.zeros(n_dof, dtype=u.dtype)
        return R.index_add(0, dofs.reshape(-1), f_e.reshape(-1))

    def tangent(u):
        # closed-form consistent tangent: sigma = s(eps_eq) C0 eps with
        # s(x) = (1 + x/eps_y)^(-1/2), so
        # D = s C0 + s'(eps_eq) (C0 eps) (d eps_eq/d eps)^T
        eps = strain(u).detach()
        eps_eq = torch.sqrt(eps[:, 0] ** 2 + eps[:, 1] ** 2 + 0.5 * eps[:, 2] ** 2 + 1e-24)
        sc = 1.0 / torch.sqrt(1.0 + eps_eq / EPS_Y)
        dsc = -0.5 / (EPS_Y * (1.0 + eps_eq / EPS_Y) ** 1.5)
        deq = torch.stack([eps[:, 0], eps[:, 1], 0.5 * eps[:, 2]], dim=1) / eps_eq.unsqueeze(1)
        C0eps = torch.einsum("ij,ej->ei", C0, eps)
        Dloc = sc.view(-1, 1, 1) * C0 + dsc.view(-1, 1, 1) * torch.einsum(
            "ei,ej->eij", C0eps, deq)                              # (n_el, 3, 3)
        K_e = torch.einsum("e,eai,eab,ebj->eij", area, B, Dloc, B)  # (n_el, 6, 6)
        K = torch.zeros(n_dof, n_dof, dtype=u.dtype)
        rows = dofs.unsqueeze(-1).expand(-1, 6, 6).reshape(-1)
        cols = dofs.unsqueeze(1).expand(-1, 6, 6).reshape(-1)
        return K.index_put((rows, cols), K_e.reshape(-1), accumulate=True)

    return residual, tangent


def run(nonlinear, max_disp=0.6, n_steps=40):
    mesh = build_double_cantilever_mesh(LENGTH, ARM, A0, NX, NY)
    law = BilinearMixedModeTSL(T_max_n=SIGMA0, T_max_s=SIGMA0, G_c1=GC, G_c2=GC, K=K0,
                               viscosity=0.1)
    kwargs = {}
    if nonlinear:
        n_dof = mesh.points.shape[0] * 2
        res_fn, tan_fn = make_nonlinear_bulk(mesh.points, mesh.elements, n_dof)
        kwargs = dict(bulk_residual_fn=res_fn, bulk_tangent_fn=tan_fn)
    model = CohesiveMeshModel(mesh.points, {"triangle": mesh.elements},
                              mesh.cohesive_connectivity, law, E=E, nu=NU, **kwargs)
    dtype = model.points.dtype
    right = model.dof_indices(mesh.right_edge_nodes)
    tt = model.dof_indices(torch.tensor([mesh.tip_top]))[1]
    tb = model.dof_indices(torch.tensor([mesh.tip_bottom]))[1]
    dofs = torch.cat([right, tt.reshape(1), tb.reshape(1)])

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)
    ds, Ps = [0.0], [0.0]
    for d in torch.linspace(0.0, max_disp, n_steps, dtype=dtype)[1:]:
        vals = torch.cat([torch.zeros(right.numel(), dtype=dtype),
                          torch.tensor([d / 2, -d / 2], dtype=dtype)])
        res = newton_solve(model, dofs, vals, kappa, u0=u, max_iter=80)
        if not res.converged:
            break
        u, kappa = res.u, res.kappa
        ds.append(float(d))
        Ps.append(0.5 * (res.reaction[-2] - res.reaction[-1]).item())
    return ds, Ps, model, mesh, dofs, u


def main():
    d_lin, P_lin, *_ = run(nonlinear=False)
    d_nl, P_nl, model, mesh, dofs, u_warm = run(nonlinear=True)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(d_lin, P_lin, "o-", ms=3, label="linear-elastic arms")
    plt.plot(d_nl, P_nl, "s-", ms=3, label="saturating-hardening arms (via bulk hooks)")
    plt.xlabel("Opening displacement (mm)")
    plt.ylabel("Reaction load P (N)")
    plt.legend()
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "nonlinear_bulk_dcb.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"peak: linear {max(P_lin):.4f}, nonlinear arms {max(P_nl):.4f}")

    # adjoint gradient with the NONLINEAR bulk active, checked against finite differences.
    # A short sweep to the check displacement provides the Newton warm start (an initial
    # guess only; the differentiated problem itself starts from the pristine history state).
    *_, model, mesh, dofs, u_warm = run(nonlinear=True, max_disp=0.15, n_steps=8)
    u_warm = u_warm.detach().clone()   # initial guess only; must carry no autograd graph
    dtype = model.points.dtype
    right_n = dofs.numel() - 2
    d = 0.15   # damage active, single-step Newton comfortable
    vals = torch.cat([torch.zeros(right_n, dtype=dtype),
                      torch.tensor([d / 2, -d / 2], dtype=dtype)])
    law = model.elem.law
    theta = theta_from_law(law).detach().requires_grad_(True)
    u_star = solve_diff(theta, model, dofs, vals, model.init_history(), u0=u_warm)
    loss = u_star.norm()
    loss.backward()
    g_adj = theta.grad.clone()

    names = [n for n, _ in law.named_parameters()]
    for pname in ("K", "G_c1"):
        i_p = names.index(pname)
        h = 1e-6 * max(abs(theta[i_p].item()), 1.0)
        f = {}
        for sgn in (1.0, -1.0):
            th = theta.detach().clone()
            th[i_p] += sgn * h
            u_p = solve_diff(th, model, dofs, vals, model.init_history(), u0=u_warm)
            f[sgn] = u_p.norm().item()
        g_fd = (f[1.0] - f[-1.0]) / (2 * h)
        denom = max(abs(g_fd), abs(g_adj[i_p].item()), 1e-12)
        rel = abs(g_adj[i_p].item() - g_fd) / denom
        print(f"d||u||/d{pname} with nonlinear bulk: adjoint {g_adj[i_p]:.6e}, "
              f"finite difference {g_fd:.6e}, relative difference {rel:.2e}")


if __name__ == "__main__":
    main()
