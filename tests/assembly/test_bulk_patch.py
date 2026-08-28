"""Uniform-strain patch tests for the TensorMesh-assembled bulk, through CohesiveMeshModel
(so the test exercises exactly the construction path users get, including the meshio->
TensorMesh node-reordering for tensor-product cells). A conforming element must reproduce a
uniform-strain state EXACTLY; a node-ordering mismatch shows up as a reaction off by 1/sqrt(3),
which is the regression this test pins down (triangles are simplex and ordering-insensitive,
so only quad/hexahedron are at risk). Tolerance reflects TensorMesh float32 assembly."""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import BilinearMixedModeTSL

E, NU = 70000.0, 0.3
LAW = dict(T_max_n=1.0, T_max_s=1.0, G_c1=1.0, G_c2=1.0, eta=1.0, K=1.0)


def _model(points, cell_type, conn):
    law = BilinearMixedModeTSL(**LAW)
    dim = points.shape[1]
    n_coh_nodes = 8 if dim == 3 else 4
    return CohesiveMeshModel(
        points=points, bulk_elements={cell_type: conn},
        cohesive_connectivity=torch.zeros(0, n_coh_nodes, dtype=torch.long),
        law=law, E=E, nu=NU,
    )


def test_quad_uniaxial_patch_exact():
    # single unit quad, meshio CCW ordering; plane strain uniaxial: u = (eps*x, -nu*eps*y)
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    conn = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    m = _model(pts, "quad", conn)
    eps = 1.0e-3
    u = torch.zeros(m.n_dof, dtype=torch.float64)
    for i in range(4):
        u[2 * i] = eps * pts[i, 0]
        u[2 * i + 1] = -NU * eps * pts[i, 1]
    F = m.K_bulk @ u
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    sigma_xx = lam * (eps - NU * eps) + 2 * mu * eps   # plane strain, eps_zz = 0
    Fx = F[2 * 1] + F[2 * 2]                            # nodes on the x = 1 face
    assert torch.isclose(Fx, torch.tensor(sigma_xx, dtype=torch.float64), rtol=1e-5)


def test_hex_uniaxial_patch_exact():
    # single unit hex, meshio/VTK ordering; 3D uniaxial with free Poisson contraction
    pts = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.float64)
    conn = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    m = _model(pts, "hexahedron", conn)
    eps = 1.0e-3
    u = torch.zeros(m.n_dof, dtype=torch.float64)
    for i in range(8):
        u[3 * i] = eps * pts[i, 0]
        u[3 * i + 1] = -NU * eps * pts[i, 1]
        u[3 * i + 2] = -NU * eps * pts[i, 2]
    F = m.K_bulk @ u
    Fx = sum(F[3 * i] for i in (1, 2, 5, 6))            # nodes on the x = 1 face
    assert torch.isclose(Fx, torch.tensor(E * eps, dtype=torch.float64), rtol=1e-5)
