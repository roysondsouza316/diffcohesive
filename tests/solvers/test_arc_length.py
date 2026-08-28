"""Arc-length validation: plain force-controlled Newton cannot pass the
snap-back peak; arc-length traces load and displacement through it to full delamination,
and its peak load must match the peak reaction found under displacement control."""

import torch

from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.solvers import newton_solve, arc_length_solve


def _cracked_model():
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    ins = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])
    law = BilinearMixedModeTSL(T_max_n=5.0, T_max_s=5.0, G_c1=0.05, G_c2=0.05, eta=1.0, K=1.0e4)
    model = CohesiveMeshModel(
        points=ins.points,
        bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity,
        law=law,
        E=1000.0,
        nu=0.3,
    )
    return model


def _force_controlled_newton_step(model, fixed_dofs, f_hat, kappa, u0, lam, tol=1e-9, max_iter=30):
    """Minimal load-controlled Newton (no arc-length) for comparison: fixed load factor lam,
    solve for u. Used only to demonstrate the snap-back failure mode, not part of the library."""
    all_dofs = torch.arange(model.n_dof)
    is_fixed = torch.zeros(model.n_dof, dtype=torch.bool)
    is_fixed[fixed_dofs] = True
    free = all_dofs[~is_fixed]
    u = u0.clone()
    for _ in range(max_iter):
        R, kappa_new, damage = model.residual(u, kappa)
        res = R[free] - lam * f_hat[free]
        if res.norm().item() < tol:
            return u, kappa_new, damage, True
        K = model.tangent(u.detach(), kappa)
        K_ff = K[free.unsqueeze(-1), free.unsqueeze(0)]
        try:
            du = torch.linalg.solve(K_ff, -res.detach())
        except torch._C._LinAlgError:
            return u, kappa, torch.zeros_like(kappa), False  # singular tangent: past the limit point
        u = u.clone()
        u[free] = u[free] + du
    return u, kappa, torch.zeros_like(kappa), False


def test_force_controlled_newton_fails_past_snapback_peak():
    model = _cracked_model()
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))
    f_hat = torch.zeros(model.n_dof, dtype=torch.float64)
    f_hat[node3_dofs[1]] = 1.0

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=torch.float64)
    diverged = False
    for lam in torch.linspace(0.5, 4.0, 20, dtype=torch.float64):
        u_try, kappa_try, damage_try, converged = _force_controlled_newton_step(
            model, fixed_dofs, f_hat, kappa, u, lam.item()
        )
        if not converged:
            diverged = True
            break
        u, kappa = u_try, kappa_try
    assert diverged, "expected plain load-controlled Newton to fail past the snap-back peak"


def test_arc_length_traces_full_snapback_to_failure():
    model = _cracked_model()
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))
    f_hat = torch.zeros(model.n_dof, dtype=torch.float64)
    f_hat[node3_dofs[1]] = 1.0

    kappa0 = model.init_history()
    history = arc_length_solve(model, fixed_dofs, f_hat, kappa0, ds=0.002, n_steps=40)

    assert all(h.converged for h in history)
    assert len(history) == 40  # traced the whole path without stalling

    lams = [h.lam for h in history]
    peak_idx = max(range(len(lams)), key=lambda i: lams[i])
    assert 0 < peak_idx < len(lams) - 1  # a genuine interior peak, i.e. we passed through it

    # Softens all the way back down past the peak, to (near) zero residual load capacity.
    assert lams[-1] < 0.05 * max(lams)
    assert history[-1].damage.max().item() > 0.99

    # Cross-check vs the displacement-controlled peak reaction (~3.13, computed
    # independently in tests/assembly/test_global_newton.py's underlying model/law).
    assert abs(max(lams) - 3.13) / 3.13 < 0.05
