"""Global assembly combining TensorMesh's native bulk elasticity with our custom cohesive
elements: the bulk needs no custom work beyond wiring -- R_coh and K_coh are added into the
global (bulk + cohesive) system.

Sparse assembly: TensorMesh's own bulk assembler already returns a native sparse
matrix; ``residual()``'s bulk term uses it directly via ``torch.sparse.mm`` instead of a dense
matvec (verified vmap-compatible, so this also benefits ``solvers.newton_batched``). ``tangent()``
stays on a *dense* global matrix, for two independent reasons, not just inertia: (1) boundary
conditions here are applied by literally slicing out the free-dof submatrix
(``K[free][:, free]``) before ``torch.linalg.solve`` -- a dense operation regardless of how K
was built -- and (2) constructing a *sparse* COO tensor from a *batched* value tensor (as
``tangent``'s per-batch-element cohesive scatter would need, under ``torch.func.vmap``) is not
supported by PyTorch today (confirmed directly: ``torch.sparse_coo_tensor`` raises
``NotImplementedError`` for the ``aten::view`` op under vmap), so making the cohesive
scatter itself sparse would silently break ``newton_solve_batched``. This is a real, tested
constraint, not a shortcut -- revisit if a future PyTorch/vmap release adds that support, or if
mesh sizes grow enough that the dense tangent's O(n_dof^2) memory becomes the bottleneck (our
validated meshes here top out around 600 DOF, where it isn't).
"""

from typing import Dict, Optional

import meshio
import numpy as np
import torch
from tensormesh.assemble import LinearElasticityElementAssembler
from tensormesh.mesh import Mesh

from ..elements.cohesive import CohesiveElement2D
from ..elements.cohesive3d import CohesiveElement3D
from ..laws.base import TractionSeparationLaw


class CohesiveMeshModel:
    def __init__(
        self,
        points: torch.Tensor,
        bulk_elements: Dict[str, torch.Tensor],
        cohesive_connectivity: torch.Tensor,
        law: TractionSeparationLaw,
        E: float = None,
        nu: float = None,
        n_quad: int = None,
        bulk_stiffness: torch.Tensor = None,
        bulk_residual_fn=None,
        bulk_tangent_fn=None,
    ):
        """``bulk_stiffness``: optional precomputed dense (n_dof, n_dof) global bulk stiffness,
        bypassing TensorMesh's isotropic assembler -- used for orthotropic/anisotropic bulk via
        assembly/orthotropic.py (TensorMesh's LinearElasticityElementAssembler is
        isotropic-only). When given, ``E``/``nu`` are ignored.

        ``bulk_residual_fn(u) -> R_bulk`` / ``bulk_tangent_fn(u) -> K_bulk(u)``: optional
        NONLINEAR bulk hooks replacing the linear ``K_bulk @ u`` term entirely -- the extension
        point for elastic-plastic or hyperelastic bulk (e.g. wrapping TensorMesh's plasticity
        machinery): the solvers and the implicit-diff adjoint only ever see the total residual
        and tangent, so a nonlinear bulk composes with the cohesive interfaces unchanged.
        Both callables must be supplied together; ``bulk_residual_fn`` must be differentiable
        w.r.t. ``u`` if adjoint gradients are wanted. Caveats:
        bulk internal variables (plastic strains) are NOT managed by this class -- the caller
        owns them, exactly as TensorMesh's own plasticity examples do between load steps --
        and no elastic-plastic validation case ships yet (the hook is verified against the
        linear path; see tests/assembly/test_bulk_hooks.py)."""
        cells = [(name, conn.detach().cpu().numpy().astype(np.int64)) for name, conn in bulk_elements.items()]
        meshio_mesh = meshio.Mesh(points=points.detach().cpu().numpy(), cells=cells)
        self.mesh = Mesh(meshio_mesh)
        self.dim = self.mesh.dim
        self.n_points = self.mesh.n_points
        self.n_dof = self.n_points * self.dim

        if (bulk_residual_fn is None) != (bulk_tangent_fn is None):
            raise ValueError("bulk_residual_fn and bulk_tangent_fn must be supplied together")
        self._bulk_residual_fn = bulk_residual_fn
        self._bulk_tangent_fn = bulk_tangent_fn
        if bulk_residual_fn is not None:
            # Nonlinear bulk: no constant matrix exists; keep zero placeholders so device
            # moves and shape queries still work.
            self.K_bulk = torch.zeros(self.n_dof, self.n_dof, dtype=self.mesh.points.dtype)
            self.K_bulk_sparse = self.K_bulk.to_sparse_coo()
        elif bulk_stiffness is not None:
            self.K_bulk = bulk_stiffness.to(self.mesh.points.dtype)
            self.K_bulk_sparse = self.K_bulk.to_sparse_coo().coalesce()
        else:
            if E is None or nu is None:
                raise ValueError("provide either (E, nu) for isotropic bulk or bulk_stiffness")
            bulk_assembler = LinearElasticityElementAssembler.from_mesh(self.mesh, E=E, nu=nu)
            K_bulk_native = bulk_assembler(self.mesh.points)
            self.K_bulk = K_bulk_native.to_dense()
            self.K_bulk_sparse = K_bulk_native.to_torch_sparse().coalesce()

        self.points = points.to(self.mesh.points.dtype)
        self.cohesive_connectivity = cohesive_connectivity
        self.n_coh = cohesive_connectivity.shape[0]
        # Element choice by spatial dimension: 4-node line-pair element in 2D, 8-node
        # face-pair element (COH3D8-equivalent) in 3D.
        if self.dim == 3:
            self.elem = CohesiveElement3D(law)
            self.n_quad = 4 if n_quad is None else n_quad
        else:
            self.elem = CohesiveElement2D(law)
            self.n_quad = 2 if n_quad is None else n_quad
        # Precomputed (n_coh, n_nodes_per_coh * dim) DOF map for the vmap-vectorized loop.
        n_coh_nodes = 8 if self.dim == 3 else 4
        if self.n_coh > 0:
            self._coh_dofs = (
                cohesive_connectivity.unsqueeze(-1) * self.dim
                + torch.arange(self.dim, dtype=torch.long)
            ).reshape(self.n_coh, -1)
        else:
            self._coh_dofs = torch.zeros(0, n_coh_nodes * self.dim, dtype=torch.long)

    def to(self, device):
        """Move the model's tensor state (and its law's parameters) to ``device``. Mesh
        construction itself (meshio/TensorMesh) is CPU-only (numpy-based), but the actual
        repeated FEM-solve arithmetic -- K_bulk, points, cohesive connectivity, the law's
        parameters -- can run on any device once moved here. ``dof_indices``/the solvers move
        any CPU-built index tensors (e.g. from mesh building) onto this device automatically,
        so callers don't need to manually port every node-id tensor themselves."""
        self.K_bulk = self.K_bulk.to(device)
        self.K_bulk_sparse = self.K_bulk_sparse.to(device)
        self.points = self.points.to(device)
        self.cohesive_connectivity = self.cohesive_connectivity.to(device)
        self._coh_dofs = self._coh_dofs.to(device)
        self.elem.law.to(device)
        return self

    def dof_indices(self, node_ids: torch.Tensor) -> torch.Tensor:
        node_ids = node_ids.to(self.points.device)
        return (node_ids.unsqueeze(-1) * self.dim + torch.arange(self.dim, device=self.points.device)).reshape(-1)

    def init_history(self) -> torch.Tensor:
        """Zero initial history; (n_coh, n_quad) for scalar-history laws, or
        (n_coh, n_quad, state_dim) for laws with extra internal variables (law.state_dim > 1,
        e.g. the frictional law's [damage history, plastic slip])."""
        state_dim = getattr(self.elem.law, "state_dim", 1)
        if state_dim == 1:
            return torch.zeros(self.n_coh, self.n_quad, dtype=self.points.dtype, device=self.points.device)
        return torch.zeros(
            self.n_coh, self.n_quad, state_dim, dtype=self.points.dtype, device=self.points.device
        )

    def residual(self, u: torch.Tensor, kappa_state: torch.Tensor, law_params: Optional[Dict[str, torch.Tensor]] = None):
        """R(u) = K_bulk @ u + sum_e R_coh^e(u), differentiable w.r.t. u (out-of-place scatter).
        The bulk term uses the native sparse matrix (torch.sparse.mm), not a dense matvec, and
        the cohesive-element loop is vectorized over elements with torch.func.vmap (identical
        math to the per-element loop it replaced; the loop was the dominant assembly cost on
        meshes with O(100) cohesive elements).

        ``law_params`` optionally substitutes the cohesive law's parameters functionally
        (for the implicit-differentiation adjoint), without mutating the live law module.
        """
        if self._bulk_residual_fn is not None:
            R = self._bulk_residual_fn(u)
        else:
            R = torch.sparse.mm(self.K_bulk_sparse, u.unsqueeze(-1)).squeeze(-1)
        if self.n_coh == 0:
            return R, kappa_state.clone(), torch.zeros_like(kappa_state)

        u_elems = u[self._coh_dofs]                     # (n_coh, 8)
        X_elems = self.points[self.cohesive_connectivity]  # (n_coh, 4, 2)

        def elem_residual(u_e, X_e, kappa_e):
            return self.elem.residual(u_e, X_e, kappa_e, law_params=law_params)

        R_e, kappa_new, damage = torch.func.vmap(elem_residual)(u_elems, X_elems, kappa_state)
        R = R.index_add(0, self._coh_dofs.reshape(-1), R_e.reshape(-1))
        return R, kappa_new, damage

    def tangent(self, u: torch.Tensor, kappa_state: torch.Tensor, law_params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """K(u) = K_bulk + sum_e K_coh^e(u); used inside Newton, not itself required to be
        differentiable (the outer implicit-diff adjoint only needs the residual).
        ``law_params`` mirrors ``residual``'s functional substitution -- needed by
        ``solvers.newton_batched`` so a vmapped Newton solve can use a per-batch-element theta
        without mutating the (shared, single) live law module. Built via an out-of-place
        ``index_put`` (like ``residual``'s ``index_add``), not in-place ``K[...] +=``, since
        vmap cannot trace an in-place write into an unbatched destination from a batched source.
        """
        if self._bulk_tangent_fn is not None:
            K = self._bulk_tangent_fn(u)
        else:
            K = self.K_bulk.clone()
        if self.n_coh == 0:
            return K

        u_elems = u[self._coh_dofs].detach()             # (n_coh, 8)
        X_elems = self.points[self.cohesive_connectivity]  # (n_coh, 4, 2)
        # Non-finite inputs can hard-crash the native vmap/jacrev/MKL kernels on Windows
        # (access violation, not a Python exception); fail loudly and catchably instead.
        # Under an outer torch.func.vmap (the batched Newton solve) data-dependent branching
        # is not traceable -- skip the check there rather than break batching.
        try:
            inputs_finite = bool(torch.isfinite(u_elems).all()) and bool(torch.isfinite(kappa_state).all())
        except RuntimeError:
            inputs_finite = True  # batched tracing: cannot branch on data
        if not inputs_finite:
            raise FloatingPointError("non-finite inputs to cohesive tangent assembly")

        def elem_tangent(u_e, X_e, kappa_e):
            return self.elem.tangent(u_e, X_e, kappa_e, law_params=law_params)

        K_e = torch.func.vmap(elem_tangent)(u_elems, X_elems, kappa_state)  # (n_coh, 8, 8)
        n = self._coh_dofs.shape[1]
        row_idx = self._coh_dofs.unsqueeze(-1).expand(-1, n, n).reshape(-1)
        col_idx = self._coh_dofs.unsqueeze(1).expand(-1, n, n).reshape(-1)
        K = K.index_put((row_idx, col_idx), K_e.reshape(-1), accumulate=True)
        return K
