"""Full 3D DCB solve through the complete stack (hex bulk via TensorMesh + 8-node cohesive
elements + Newton): elastic compliance must match beam theory for the narrow free-sided 3D
specimen (C = 8a^3/(E w h^3), plain E -- free lateral faces, unlike the plane-strain 2D model),
damage must initiate and grow at the crack tip, and the response must peak and soften."""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import build_double_cantilever_mesh_3d
from diffcohesive.solvers import arc_length_solve, newton_solve

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


def test_3d_dcb_elastic_compliance_converges_to_beam_theory():
    """Trilinear (Q1) hexes shear-lock in slender bending, so a coarse mesh is markedly too
    stiff -- the correct expectation is CONVERGENCE toward beam theory under refinement, which
    is what this asserts (plus a done-converging bound on the finer mesh)."""
    C_beam = 8.0 * A0 ** 3 / (E * W * ARM ** 3)  # plain E: narrow beam, free lateral faces
    C_coarse = _elastic_compliance(30, 2, 1)
    C_fine = _elastic_compliance(60, 4, 1)
    err_coarse = abs(C_coarse - C_beam) / C_beam
    err_fine = abs(C_fine - C_beam) / C_beam
    assert err_fine < err_coarse  # refinement moves toward beam theory
    assert err_fine < 0.25


def test_3d_dcb_damage_initiates_peaks_and_softens():
    model, mesh = _model()
    assert model.dim == 3 and model.n_quad == 4
    dtype = model.points.dtype
    kappa = model.init_history()
    assert kappa.shape == (model.n_coh, 4)
    u = torch.zeros(model.n_dof, dtype=dtype)

    # Displacement-controlled Newton up to the first cohesive pop-in (a local limit point,
    # exactly as in 2D)...
    P, max_damage = [], []
    u_prev_norm = None
    for d in torch.linspace(0.0, 1.6, 26, dtype=dtype):
        result, P_top = _solve_opening(model, mesh, d.item(), kappa, u)
        if not result.converged:
            break
        u_prev_norm = (result.u - u).norm().item()
        u, kappa = result.u, result.kappa
        P.append(P_top)
        max_damage.append(result.damage.max().item())

    assert max_damage[0] < 1e-9
    assert max_damage[-1] > 0.9  # first elements essentially failed before the pop-in

    # ...then adaptive arc-length through the pop-in traces the softening branch (the same
    # handoff validated at length in 2D; this exercises it through the 3D element stack).
    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    top_y = model.dof_indices(mesh.tip_top_nodes).reshape(-1, 3)[:, 1]
    bot_y = model.dof_indices(mesh.tip_bottom_nodes).reshape(-1, 3)[:, 1]
    f_hat = torch.zeros(model.n_dof, dtype=dtype)
    f_hat[top_y] = 1.0 / top_y.numel()
    f_hat[bot_y] = -1.0 / bot_y.numel()
    history = arc_length_solve(
        model, fixed_dofs=right_dofs, f_hat=f_hat, kappa_state=kappa,
        ds=0.5 * (u_prev_norm or 0.1), n_steps=25, u0=u, lam0=P[-1],
    )
    lams = [s.lam for s in history if s.converged]
    assert len(lams) >= 5
    peak = max(P + lams)
    assert lams[-1] < peak  # softening past the peak, through the 3D cohesive stack
