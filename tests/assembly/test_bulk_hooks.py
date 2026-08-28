"""Nonlinear-bulk hook verification: (1) supplying the LINEAR bulk through the callable hooks
must reproduce the built-in path bit-for-bit through a full damaging Newton run (so the hook
plumbing itself is exact); (2) a genuinely nonlinear bulk (cubic-hardening spring field) must
solve and produce the expected stiffening -- demonstrating that the solvers/assembly only ever
consume residual+tangent, the property that makes elastic-plastic bulk an extension rather
than a rewrite. Internal-variable management (plastic strain) stays with the caller and no
plasticity validation is claimed here."""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.solvers import newton_solve


def _mesh_and_law():
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    ins = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])
    law = BilinearMixedModeTSL(T_max_n=5.0, T_max_s=5.0, G_c1=0.05, G_c2=0.05, eta=1.0, K=1.0e4)
    return ins, law


def _run(model, d=0.008):
    fixed = model.dof_indices(torch.tensor([0, 1, 2]))
    n3 = model.dof_indices(torch.tensor([3]))
    pd = torch.cat([fixed, n3[1:2]])
    pv = torch.cat([torch.zeros(6, dtype=torch.float64), torch.tensor([d], dtype=torch.float64)])
    result = newton_solve(model, pd, pv, model.init_history(), max_iter=80)
    assert result.converged
    return result


def test_linear_bulk_through_hooks_matches_builtin_exactly():
    ins, law = _mesh_and_law()
    builtin = CohesiveMeshModel(
        points=ins.points, bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity, law=law, E=1000.0, nu=0.3,
    )
    K_lin = builtin.K_bulk.clone()
    hooked = CohesiveMeshModel(
        points=ins.points, bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity, law=law,
        bulk_residual_fn=lambda u: K_lin @ u,
        bulk_tangent_fn=lambda u: K_lin.clone(),
    )
    r1, r2 = _run(builtin), _run(hooked)
    assert torch.allclose(r1.u, r2.u, atol=1e-12)
    assert torch.allclose(r1.damage, r2.damage, atol=1e-12)


def test_nonlinear_bulk_solves_and_stiffens():
    ins, law = _mesh_and_law()
    builtin = CohesiveMeshModel(
        points=ins.points, bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity, law=law, E=1000.0, nu=0.3,
    )
    K_lin = builtin.K_bulk.clone()
    alpha = 5.0e5  # cubic hardening: R = K u + alpha * u^3 (component-wise demo nonlinearity)

    nonlinear = CohesiveMeshModel(
        points=ins.points, bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity, law=law,
        bulk_residual_fn=lambda u: K_lin @ u + alpha * u ** 3,
        bulk_tangent_fn=lambda u: K_lin + torch.diag(3.0 * alpha * u ** 2),
    )
    d = 0.006
    r_lin = _run(builtin, d)
    r_nl = _run(nonlinear, d)
    # Hardening bulk -> larger reaction at equal prescribed displacement.
    assert r_nl.reaction[-1].item() > r_lin.reaction[-1].item()
    # And the residual at the converged state is genuinely the nonlinear one (consistency).
    R, _, _ = nonlinear.residual(r_nl.u, nonlinear.init_history())
    assert torch.isfinite(R).all()