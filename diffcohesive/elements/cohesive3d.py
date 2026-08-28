"""3D 8-node zero-thickness cohesive element (COH3D8-equivalent): a pair of coincident
bilinear quadrilateral faces. Node convention: nodes 0-3 = bottom face (cyclic), nodes 4-7 =
top face with node i+4 coincident with node i in the reference configuration. Element DOFs
u_elem are (24,) ordered [n0x, n0y, n0z, n1x, ...].

Kinematics: the separation at a face point is Delta_u = sum_i N_i (u_top_i - u_bot_i), rotated
to the local frame [n, t1, t2] built from the reference face tangents, giving
delta_local = (delta_n, delta_s1, delta_s2) -- exactly the (..., ndim) layout the
traction-separation laws already accept (they reduce the shear components to an effective
shear magnitude internally).

Integration: 2x2 Gauss on the face. Tangent: forward-mode autograd (jacfwd), matching the 2D
element; never hand-derived. The local frame is computed from the *reference* geometry
(small-rotation assumption, same as the 2D element)."""

import math
from typing import Dict, Optional

import torch

from ..laws.base import TractionSeparationLaw

_G = 1.0 / math.sqrt(3.0)
# 2x2 Gauss points on the reference face (xi, eta), all weights 1.
_GAUSS4 = torch.tensor(
    [[-_G, -_G], [_G, -_G], [_G, _G], [-_G, _G]], dtype=torch.float64
)


def _shape_functions(xi, eta):
    """Bilinear quad shape functions and derivatives, node order (-,-), (+,-), (+,+), (-,+)."""
    N = torch.stack(
        [
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ]
    )
    dN_dxi = torch.stack([-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)])
    dN_deta = torch.stack([-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)])
    return N, dN_dxi, dN_deta


class CohesiveElement3D:
    def __init__(self, law: TractionSeparationLaw, quad_points: torch.Tensor = _GAUSS4):
        self.law = law
        self.quad_points = quad_points
        self.n_quad = quad_points.shape[0]

    def residual(
        self,
        u_elem: torch.Tensor,       # (24,)
        X_elem: torch.Tensor,       # (8, 3) reference coords; top nodes coincide with bottom
        kappa_prev: torch.Tensor,   # (n_quad,) or (n_quad, state_dim)
        law_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        u = u_elem.reshape(8, 3)
        du_nodes = u[4:] - u[:4]              # (4, 3) top-minus-bottom per node pair
        X_face = X_elem[:4]                    # (4, 3) reference (coincident) face

        R_e = torch.zeros(24, dtype=u_elem.dtype, device=u_elem.device)
        kappa_new_list, damage_list = [], []
        for q in range(self.n_quad):
            xi = self.quad_points[q, 0].to(u_elem.device)
            eta = self.quad_points[q, 1].to(u_elem.device)
            N, dN_dxi, dN_deta = _shape_functions(xi, eta)

            t_xi = (dN_dxi.unsqueeze(-1) * X_face).sum(0)    # (3,)
            t_eta = (dN_deta.unsqueeze(-1) * X_face).sum(0)  # (3,)
            n_vec = torch.linalg.cross(t_xi, t_eta)
            jac = n_vec.norm()
            n_hat = n_vec / jac
            t1 = t_xi / t_xi.norm()
            t2 = torch.linalg.cross(n_hat, t1)
            R_mat = torch.stack([n_hat, t1, t2])             # (3, 3): local = R @ global

            delta_u = (N.unsqueeze(-1) * du_nodes).sum(0)    # (3,)
            delta_local = R_mat @ delta_u

            if law_params is None:
                traction, kappa_q, damage_q = self.law(delta_local, kappa_prev[q])
            else:
                traction, kappa_q, damage_q = torch.func.functional_call(
                    self.law, law_params, (delta_local, kappa_prev[q])
                )

            t_global = R_mat.transpose(0, 1) @ traction      # (3,) traction in global frame
            # d(delta_u)/du: -N_i on bottom nodes, +N_i on top nodes.
            contrib = (N.unsqueeze(-1) * t_global.unsqueeze(0)).reshape(-1) * jac  # (12,)
            R_e = R_e + torch.cat([-contrib, contrib])
            kappa_new_list.append(kappa_q)
            damage_list.append(damage_q)

        return R_e, torch.stack(kappa_new_list), torch.stack(damage_list)

    def tangent(
        self,
        u_elem: torch.Tensor,
        X_elem: torch.Tensor,
        kappa_prev: torch.Tensor,
        law_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """K_coh^e = dR_e/du_elem (24 x 24) via forward-mode autograd."""

        def residual_only(u):
            R_e, _, _ = self.residual(u, X_elem, kappa_prev, law_params=law_params)
            return R_e

        return torch.func.jacfwd(residual_only)(u_elem)
