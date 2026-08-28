"""Orthotropic linear-elastic bulk stiffness for 2D meshes (closing the isotropy-only
limitation): a pure-torch constant-strain-triangle (CST) assembler with an orthotropic
plane-strain or plane-stress constitutive matrix, optionally rotated by a material (fiber)
angle. Used through CohesiveMeshModel's ``bulk_stiffness`` override, bypassing TensorMesh's
isotropic-only ``LinearElasticityElementAssembler`` for the bulk while keeping everything else
(cohesive elements, solvers, adjoint) unchanged.

Constitutive matrix: the full 3D orthotropic compliance is built from engineering constants
(unidirectional-ply convention: 1 = fiber/x, 2 = in-plane transverse/y, 3 = out-of-plane),
inverted to the 3D stiffness, and reduced to the plane problem:

- plane strain (eps_zz = gamma_xz = gamma_yz = 0): the [xx, yy, xy] stiffness submatrix;
- plane stress: the classical reduced stiffness Q from the in-plane compliance.

For the isotropic limit (E1 = E2 = E3, all nu equal, G = E/(2(1+nu))) the plane-strain matrix
reduces exactly to TensorMesh's isotropic plane-strain operator -- asserted in the tests by
comparing the assembled global matrices entry-wise on the same mesh.

The out-of-plane constants default to the transverse ones (E3 = E2, nu13 = nu12, G23 = G12),
the usual transversely-isotropic approximation for a unidirectional ply; override them if
measured values are available.
"""

import math
from typing import Optional

import torch


def orthotropic_plane_C(
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    E3: Optional[float] = None,
    nu13: Optional[float] = None,
    nu23: Optional[float] = None,
    G23: Optional[float] = None,
    state: str = "plane_strain",
    theta_deg: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """(3, 3) constitutive matrix relating (eps_xx, eps_yy, gamma_xy) -> (sig_xx, sig_yy, tau_xy),
    with the material 1-axis rotated by ``theta_deg`` from the global x-axis."""
    if state not in ("plane_strain", "plane_stress"):
        raise ValueError(f"state must be 'plane_strain' or 'plane_stress', got {state!r}")
    E3 = E2 if E3 is None else E3
    nu13 = nu12 if nu13 is None else nu13
    nu23 = nu12 if nu23 is None else nu23
    G23 = G12 if G23 is None else G23

    if state == "plane_stress":
        nu21 = nu12 * E2 / E1
        denom = 1.0 - nu12 * nu21
        C = torch.tensor(
            [
                [E1 / denom, nu12 * E2 / denom, 0.0],
                [nu12 * E2 / denom, E2 / denom, 0.0],
                [0.0, 0.0, G12],
            ],
            dtype=dtype,
        )
    else:
        # Full orthotropic 3D compliance (Voigt order xx, yy, zz, yz, xz, xy), inverted, then
        # the plane-strain stiffness submatrix [xx, yy, xy].
        S = torch.zeros(6, 6, dtype=dtype)
        S[0, 0], S[1, 1], S[2, 2] = 1.0 / E1, 1.0 / E2, 1.0 / E3
        S[0, 1] = S[1, 0] = -nu12 / E1
        S[0, 2] = S[2, 0] = -nu13 / E1
        S[1, 2] = S[2, 1] = -nu23 / E2
        S[3, 3] = 1.0 / G23
        S[4, 4] = 1.0 / G12  # G13 ~ G12 for a unidirectional ply
        S[5, 5] = 1.0 / G12
        C6 = torch.linalg.inv(S)
        idx = torch.tensor([0, 1, 5])
        C = C6[idx.unsqueeze(-1), idx.unsqueeze(0)]

    if theta_deg != 0.0:
        c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
        # Voigt (engineering shear strain) rotation: C' = T_sigma^{-1} C T_eps, with the
        # standard reserved transformations.
        T_sig_inv = torch.tensor(
            [
                [c * c, s * s, -2 * c * s],
                [s * s, c * c, 2 * c * s],
                [c * s, -c * s, c * c - s * s],
            ],
            dtype=dtype,
        )
        T_eps = torch.tensor(
            [
                [c * c, s * s, c * s],
                [s * s, c * c, -c * s],
                [-2 * c * s, 2 * c * s, c * c - s * s],
            ],
            dtype=dtype,
        )
        C = T_sig_inv @ C @ T_eps
    return C


def assemble_cst_stiffness(
    points: torch.Tensor,     # (n_points, 2)
    triangles: torch.Tensor,  # (n_tri, 3)
    C: torch.Tensor,          # (3, 3) plane constitutive matrix
) -> torch.Tensor:
    """Dense global stiffness for 3-node constant-strain triangles with unit thickness:
    K_e = A_e * B_e^T C B_e with the standard constant B, scatter-accumulated. Vectorized over
    elements."""
    n_points = points.shape[0]
    n_dof = 2 * n_points
    tri_pts = points[triangles]                # (m, 3, 2)
    x = tri_pts[:, :, 0]
    y = tri_pts[:, :, 1]

    # b_i = y_j - y_k, c_i = x_k - x_j (cyclic), 2A = x_21*y_31 - x_31*y_21
    b = torch.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], dim=1)  # (m,3)
    c = torch.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], dim=1)
    twoA = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])

    m = triangles.shape[0]
    B = torch.zeros(m, 3, 6, dtype=points.dtype)
    for i in range(3):
        B[:, 0, 2 * i] = b[:, i]
        B[:, 1, 2 * i + 1] = c[:, i]
        B[:, 2, 2 * i] = c[:, i]
        B[:, 2, 2 * i + 1] = b[:, i]
    B = B / twoA.abs().reshape(-1, 1, 1)

    area = 0.5 * twoA.abs()
    K_e = torch.einsum("mki,kl,mlj->mij", B, C, B) * area.reshape(-1, 1, 1)  # (m, 6, 6)

    dofs = (triangles.unsqueeze(-1) * 2 + torch.arange(2, dtype=torch.long)).reshape(m, 6)
    row = dofs.unsqueeze(-1).expand(m, 6, 6).reshape(-1)
    col = dofs.unsqueeze(1).expand(m, 6, 6).reshape(-1)
    K = torch.zeros(n_dof, n_dof, dtype=points.dtype)
    K = K.index_put((row, col), K_e.reshape(-1), accumulate=True)
    return K
