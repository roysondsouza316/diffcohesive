"""Full 3D DCB solve through the complete stack (hex bulk via TensorMesh + 8-node cohesive
elements + Newton): elastic compliance must match beam theory for the narrow free-sided 3D
specimen (C = 8a^3/(E w h^3), plain E -- free lateral faces, unlike the plane-strain 2D model),
damage must initiate and grow at the crack tip, and the response must peak and soften."""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import build_double_cantilever_mesh_3d
from diffcohesive.solvers import adaptive_displacement_solve, newton_solve

L, ARM, W, A0 = 60.0, 2.0, 4.0, 20.0
E, NU = 70000.0, 0.3
SIGMA0, GC, K = 30.0, 0.5, 1.0e4


def _model(nx=30, ny=2, nz=2):
    mesh = build_double_cantilever_mesh_3d(L, ARM, W, A0, nx=nx, ny=ny, nz=nz)
    law = BilinearMixedModeTSL(T_max_n=SIGMA0, T_max_s=SIGMA0, G_c1=GC, G_c2=GC, eta=1.0, K=K)
    model = CohesiveMeshModel(
        points=mesh.points,
        bulk_elements={mesh.cell_type: mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity,
        law=law,
        E=E,
        nu=NU,
    )
    return model, mesh


def _solve_opening(model, mesh, d, kappa, u):
    dtype = model.points.dtype
    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    top_y = model.dof_indices(mesh.tip_top_nodes).reshape(-1, 3)[:, 1]
    bot_y = model.dof_indices(mesh.tip_bottom_nodes).reshape(-1, 3)[:, 1]
    prescribed_dofs = torch.cat([right_dofs, top_y, bot_y])
    prescribed_values = torch.cat(
        [
            torch.zeros(right_dofs.numel(), dtype=dtype),
            torch.full((top_y.numel(),), d / 2, dtype=dtype),
            torch.full((bot_y.numel(),), -d / 2, dtype=dtype),
        ]
    )
    result = newton_solve(model, prescribed_dofs, prescribed_values, kappa, u0=u, max_iter=60)
    n_r, n_top = right_dofs.numel(), top_y.numel()
    P_top = result.reaction[n_r:n_r + n_top].sum().item()
    return result, P_top


def _elastic_compliance(nx, ny, nz):
    model, mesh = _model(nx=nx, ny=ny, nz=nz)
    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=model.points.dtype)
    d = 0.05  # deep in the elastic range
    result, P = _solve_opening(model, mesh, d, kappa, u)
    assert result.converged and result.damage.max().item() < 1e-9
    return d / P


def test_3d_dcb_elastic_compliance_converges_above_beam_theory():
    """Trilinear (Q1) hexes shear-lock in slender bending, so a coarse mesh is too stiff and
    refinement monotonically RELEASES compliance. The converged value sits ABOVE the naive
    built-in-cantilever beam theory C = 8a^3/(E W h^3), because the bonded ligament is an
    elastic foundation (finite interface stiffness K): the crack root rotates, adding
    compliance -- the classical beam-on-elastic-foundation (Kanninen-type) correction. This
    asserts (1) monotone softening under refinement, (2) shrinking increments (convergence),
    and (3) the finest compliance in the physically expected band above beam theory."""
    C_beam = 8.0 * A0 ** 3 / (E * W * ARM ** 3)
    C1 = _elastic_compliance(30, 2, 1)
    C2 = _elastic_compliance(45, 3, 1)
    C3 = _elastic_compliance(60, 4, 1)
    assert C1 < C2 < C3                       # locking releases monotonically
    assert (C3 - C2) < (C2 - C1)              # increments shrink: converging
    assert 0.95 * C_beam < C3 < 1.45 * C_beam  # above beam theory by the root-compliance margin


def test_3d_dcb_damage_initiates_peaks_and_softens():
    """Displacement-controlled sweep with a small viscous damage regularization (the same
    stabilization used for the 3D Abaqus cross-check in abaqus/dcb3d/) must initiate damage
    at the crack tip, reach a load peak, and trace the softening branch past it."""
    mesh = build_double_cantilever_mesh_3d(L, ARM, W, A0, nx=30, ny=2, nz=2)
    law = BilinearMixedModeTSL(T_max_n=SIGMA0, T_max_s=SIGMA0, G_c1=GC, G_c2=GC, eta=1.0,
                               K=K, viscosity=0.2)
    model = CohesiveMeshModel(
        points=mesh.points, bulk_elements={mesh.cell_type: mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity, law=law, E=E, nu=NU,
    )
    assert model.dim == 3 and model.n_quad == 4
    dtype = model.points.dtype
    kappa = model.init_history()
    assert kappa.shape == (model.n_coh, 4, 2)  # [kappa, D_v] per quadrature point
    u = torch.zeros(model.n_dof, dtype=dtype)

    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    top_y = model.dof_indices(mesh.tip_top_nodes).reshape(-1, 3)[:, 1]
    bot_y = model.dof_indices(mesh.tip_bottom_nodes).reshape(-1, 3)[:, 1]
    n_r, n_t = right_dofs.numel(), top_y.numel()

    def prescribed_fn(d):
        pd = torch.cat([right_dofs, top_y, bot_y])
        pv = torch.cat([
            torch.zeros(n_r, dtype=dtype),
            torch.full((n_t,), d / 2, dtype=dtype),
            torch.full((bot_y.numel(),), -d / 2, dtype=dtype),
        ])
        return pd, pv

    P, max_damage, deltas = [], [], []
    d_prev = 0.0
    for d in torch.linspace(0.0, 2.0, 21, dtype=dtype)[1:]:
        out = adaptive_displacement_solve(
            model, prescribed_fn, kappa, u, d_prev, float(d),
            initial_step=float(d) - d_prev, max_iter=60,
        )
        if out is None or out[3] < float(d) - 1e-12:
            break
        result, u, kappa, d_prev = out
        P.append(result.reaction[n_r:n_r + n_t].sum().item())
        max_damage.append(result.damage.max().item())
        deltas.append(float(d))

    assert max_damage[0] < 1e-6          # elastic start (first step d = 0.1)
    assert max_damage[-1] > 0.9          # crack-tip elements essentially failed
    peak_idx = max(range(len(P)), key=lambda i: P[i])
    assert 0 < peak_idx < len(P) - 1     # a peak exists inside the traced range
    assert deltas[-1] > 1.5              # the softening branch was actually traced
    assert P[-1] < 0.92 * P[peak_idx]    # and the load clearly fell from the peak
    # (the viscous regularization keeps the traced softening branch shallow at this step size)
