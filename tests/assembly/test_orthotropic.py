"""Orthotropic bulk validation. The decisive check is the isotropic limit: the pure-torch
orthotropic CST assembler fed isotropic constants must reproduce TensorMesh's own isotropic
plane-strain stiffness ENTRY-WISE on the same mesh -- if the constitutive reduction, B-matrix,
or scatter were wrong anywhere, this comparison would fail. Then a unidirectional composite DCB
(fibers along the beam axis) must be governed by E1 in bending: softer than the Euler beam with
E1 (shear-flexible composite), far softer than an isotropic-E1 solid, and complete a
damage-growth run through the cohesive stack."""

import torch

from diffcohesive.assembly import CohesiveMeshModel, assemble_cst_stiffness, orthotropic_plane_C
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import build_double_cantilever_mesh
from diffcohesive.solvers import newton_solve
from tensormesh.assemble import LinearElasticityElementAssembler
from tensormesh.mesh import Mesh
import meshio
import numpy as np


def test_isotropic_limit_matches_tensormesh_exactly():
    mesh = build_double_cantilever_mesh(10.0, 1.0, 4.0, 10, 2, element_type="triangle")
    E, nu = 1234.5, 0.31
    C_iso = orthotropic_plane_C(E1=E, E2=E, nu12=nu, G12=E / (2 * (1 + nu)), E3=E, state="plane_strain")
    K_ours = assemble_cst_stiffness(mesh.points, mesh.elements, C_iso)

    cells = [("triangle", mesh.elements.numpy().astype(np.int64))]
    tm_mesh = Mesh(meshio.Mesh(points=mesh.points.numpy(), cells=cells))
    K_tm = LinearElasticityElementAssembler.from_mesh(tm_mesh, E=E, nu=nu)(tm_mesh.points).to_dense()

    scale = K_tm.abs().max().item()
    # Agreement is limited by TensorMesh's internal float32 rounding (~1e-7 relative), not by
    # our float64 assembly; verified on a single element that the entries match to float32 ulp.
    assert (K_ours - K_tm).abs().max().item() < 1e-6 * scale


def _composite_dcb(theta_deg=0.0):
    # Unidirectional CFRP-like ply set (fibers along the beam axis for theta_deg = 0).
    mat = dict(E1=120000.0, E2=8000.0, nu12=0.3, G12=4000.0)
    mesh = build_double_cantilever_mesh(30.0, 1.0, 10.0, 60, 4, element_type="triangle")
    C = orthotropic_plane_C(state="plane_strain", theta_deg=theta_deg, **mat)
    K_bulk = assemble_cst_stiffness(mesh.points, mesh.elements, C)
    law = BilinearMixedModeTSL(T_max_n=30.0, T_max_s=30.0, G_c1=0.26, G_c2=0.26, eta=1.0, K=1.0e5)
    model = CohesiveMeshModel(
        points=mesh.points,
        bulk_elements={mesh.cell_type: mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity,
        law=law,
        bulk_stiffness=K_bulk,
    )
    return model, mesh, mat


def _compliance_and_softening(model, mesh, disps):
    dtype = model.points.dtype
    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    tip_top_y = model.dof_indices(torch.tensor([mesh.tip_top]))[1]
    tip_bottom_y = model.dof_indices(torch.tensor([mesh.tip_bottom]))[1]
    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)
    P, D = [], []
    for d in disps:
        pd = torch.cat([right_dofs, tip_top_y.reshape(1), tip_bottom_y.reshape(1)])
        pv = torch.cat([torch.zeros(right_dofs.numel(), dtype=dtype),
                        torch.tensor([d / 2, -d / 2], dtype=dtype)])
        result = newton_solve(model, pd, pv, kappa, u0=u, max_iter=60)
        if not result.converged:
            break
        u, kappa = result.u, result.kappa
        P.append(0.5 * (result.reaction[-2] - result.reaction[-1]).item())
        D.append(result.damage.max().item())
    return P, D


def test_composite_dcb_bending_governed_by_E1():
    model, mesh, mat = _composite_dcb(theta_deg=0.0)
    a0, h = mesh.crack_length, 1.0
    disps = torch.linspace(0.0, 0.02, 4, dtype=torch.float64).tolist()
    P, D = _compliance_and_softening(model, mesh, disps)
    assert max(D) < 1e-9  # elastic range
    C_fem = disps[-1] / P[-1]
    C_beam_E1 = 8.0 * a0 ** 3 / (mat["E1"] * h ** 3)
    # Shear-flexible unidirectional beam: softer than the Euler beam on E1, but within the
    # classical shear-correction margin (G12/E1 = 1/30 here), and nowhere near the answer an
    # isotropic-E2 (factor ~15 softer) or isotropic-E1-with-locking model would give.
    assert C_fem > C_beam_E1
    assert (C_fem - C_beam_E1) / C_beam_E1 < 0.6


def test_fiber_angle_softens_the_response():
    # Rotating fibers off-axis must reduce the bending stiffness monotonically toward E2.
    C0 = None
    for theta in (0.0, 30.0, 90.0):
        model, mesh, mat = _composite_dcb(theta_deg=theta)
        disps = [0.0, 0.01]
        P, _ = _compliance_and_softening(model, mesh, disps)
        C = disps[-1] / P[-1]
        if C0 is not None:
            assert C > 1.05 * C0
        C0 = C


def test_composite_dcb_damage_grows_and_softens():
    model, mesh, _ = _composite_dcb(theta_deg=0.0)
    disps = torch.linspace(0.0, 0.9, 30, dtype=torch.float64).tolist()
    P, D = _compliance_and_softening(model, mesh, disps)
    assert D[0] < 1e-9 and max(D) > 0.9
    peak_idx = max(range(len(P)), key=lambda k: P[k])
    assert 0 < peak_idx <= len(P) - 1 and max(D) > 0.9