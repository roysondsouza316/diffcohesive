"""2D linear (4-node) cohesive element: kinematics, residual, and autograd tangent.

Node ordering (matching diffcohesive.mesh.insertion.insert_cohesive_interface's cohesive
connectivity [a_bottom, b_bottom, b_top, a_top]): n1, n2 on the bottom face; n3, n4 on the
top face, with n4 coincident with n1 and n3 coincident with n2 in the reference (undeformed,
zero-thickness) configuration. DOF ordering in ``u_elem`` is [u1x,u1y, u2x,u2y, u3x,u3y, u4x,u4y].

The local (normal/tangential) frame is computed once from the reference bottom-edge geometry.
This assumes small bulk rotations across a load step (consistent with the package's small-strain
bulk-material scope) rather than re-computing the frame from the deformed midplane every
iteration; revisit if large rigid-body rotation of the interface becomes relevant.
"""

from typing import Optional, Tuple, Dict

import torch

from ..laws.base import TractionSeparationLaw

_GAUSS2_XI = torch.tensor([-1.0 / 3.0 ** 0.5, 1.0 / 3.0 ** 0.5], dtype=torch.float64)
_GAUSS2_W = torch.tensor([1.0, 1.0], dtype=torch.float64)


def _shape_functions(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)


def _local_frame(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """Rotation matrix R (2,2) s.t. delta_local = R @ delta_u_global = [delta_n, delta_s]."""
    edge = X2 - X1
    length = torch.linalg.norm(edge)
    t = edge / length
    n = torch.stack([-t[1], t[0]])
    return torch.stack([n, t], dim=0), length


def _relative_displacement_B(xi: torch.Tensor) -> torch.Tensor:
    """B(xi): (2,8) matrix mapping u_elem -> Delta_u = u_top(xi) - u_bottom(xi)."""
    N1, N2 = _shape_functions(xi)
    zero = torch.zeros_like(N1)
    row_x = torch.stack([-N1, zero, -N2, zero, N2, zero, N1, zero])
    row_y = torch.stack([zero, -N1, zero, -N2, zero, N2, zero, N1])
    return torch.stack([row_x, row_y], dim=0)


class CohesiveElement2D:
    def __init__(self, law: TractionSeparationLaw, quad_xi: torch.Tensor = _GAUSS2_XI, quad_w: torch.Tensor = _GAUSS2_W):
        self.law = law
        self.quad_xi = quad_xi
        self.quad_w = quad_w

    def residual(
        self,
        u_elem: torch.Tensor,
        X_elem: torch.Tensor,
        kappa_prev: torch.Tensor,
        law_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            u_elem: (8,) nodal displacements [u1x,u1y,u2x,u2y,u3x,u3y,u4x,u4y].
            X_elem: (4,2) reference nodal coordinates [X1,X2,X3,X4].
            kappa_prev: (n_quad,) history variable at each quadrature point.
            law_params: optional functional-call substitution for the law's own parameters
                (lets the implicit-diff backward pass build a fresh graph
                w.r.t. a theta leaf without mutating the live module).

        Returns:
            R_e: (8,) element residual (internal nodal forces).
            kappa_new: (n_quad,) updated history.
            damage: (n_quad,) damage at each quadrature point.
        """
        X1, X2 = X_elem[0], X_elem[1]
        R, length = _local_frame(X1, X2)

        R_e = torch.zeros(8, dtype=u_elem.dtype, device=u_elem.device)
        kappa_new_list = []
        damage_list = []
        for q, (xi, w) in enumerate(zip(self.quad_xi, self.quad_w)):
            # quad_xi/quad_w are fixed CPU constants by default; move to the element's own
            # device so this works when u_elem/X_elem live on GPU.
            xi = xi.to(u_elem.device)
            w = w.to(u_elem.device)
            B = _relative_displacement_B(xi)
            delta_u_global = B @ u_elem
            delta_local = R @ delta_u_global
            # law expects (..., ndim) with ndim = 1 normal + shear components; here 1 shear comp.
            if law_params is None:
                traction, kappa_q, damage_q = self.law(delta_local, kappa_prev[q])
            else:
                traction, kappa_q, damage_q = torch.func.functional_call(
                    self.law, law_params, (delta_local, kappa_prev[q])
                )
            jacobian = length / 2.0
            R_e = R_e + w * jacobian * (B.transpose(0, 1) @ (R.transpose(0, 1) @ traction))
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
        """K_coh^e = dR_e/du_elem via autograd -- deliberately not hand-derived.

        Forward-mode (jacfwd) rather than reverse-mode (jacrev): for this square 8->8 Jacobian
        the cost is equivalent, and the reverse-mode kernel path deterministically
        access-violates (native Windows crash, no Python exception) at certain late-propagation
        states when composed under torch.func.vmap in the global assembly -- observed with
        finite inputs and single-threaded MKL. Forward mode exercises a different native path.
        """

        def residual_only(u):
            R_e, _, _ = self.residual(u, X_elem, kappa_prev, law_params=law_params)
            return R_e

        return torch.func.jacfwd(residual_only)(u_elem)
