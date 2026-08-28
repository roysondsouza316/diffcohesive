"""Critical adjoint gate: finite-difference check of dL/dtheta vs the
implicit-diff adjoint, for every theta component. Loss = reaction force at the prescribed dofs
(the physically meaningful quantity §7 calibrates against), evaluated with theta entering both
directly (functional_call) and indirectly (through the converged u*), so the test exercises the
adjoint's full chain-rule contribution, not just a partial derivative.
"""

import torch

from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.solvers import newton_solve
from diffcohesive.diff import solve_diff, theta_from_law, split_theta, assign_theta_


def _build_model():
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


def _reaction_loss(theta, model, prescribed_dofs, prescribed_values, kappa_state, u0):
    u_star = solve_diff(theta, model, prescribed_dofs, prescribed_values, kappa_state, u0=u0)
    law_params = split_theta(model.elem.law, theta)
    R, _, _ = model.residual(u_star, kappa_state, law_params=law_params)
    # Sum over *all* prescribed dofs is ~0 by global equilibrium (whatever the free-dof
    # reactions are, they cancel against it) regardless of theta -- degenerate. Use the one
    # physically loaded reaction (node 3's prescribed dof, the last entry) instead.
    return R[prescribed_dofs[-1]]


def test_adjoint_gradient_matches_finite_difference_every_theta_component():
    model = _build_model()
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])

    # A displacement squarely in the softening regime (nontrivial, well-conditioned tangent).
    d = 0.008
    prescribed_values = torch.cat([torch.zeros(6, dtype=torch.float64), torch.tensor([d], dtype=torch.float64)])
    kappa_state = model.init_history()
    u0 = torch.zeros(model.n_dof, dtype=torch.float64)

    theta0 = theta_from_law(model.elem.law)
    theta = theta0.clone().requires_grad_(True)

    loss = _reaction_loss(theta, model, prescribed_dofs, prescribed_values, kappa_state, u0)
    loss.backward()
    grad_autograd = theta.grad.clone()
    assert torch.isfinite(grad_autograd).all()

    param_names = [name for name, _ in model.elem.law.named_parameters()]

    grad_fd = torch.zeros_like(theta0)
    for i in range(theta0.numel()):
        h = max(1e-6, 1e-5 * abs(theta0[i].item()))

        theta_plus = theta0.clone()
        theta_plus[i] += h
        assign_theta_(model.elem.law, theta_plus)
        result_plus = newton_solve(model, prescribed_dofs, prescribed_values, kappa_state, u0=u0)
        assert result_plus.converged
        R_plus, _, _ = model.residual(result_plus.u, kappa_state)
        loss_plus = R_plus[prescribed_dofs[-1]].item()

        theta_minus = theta0.clone()
        theta_minus[i] -= h
        assign_theta_(model.elem.law, theta_minus)
        result_minus = newton_solve(model, prescribed_dofs, prescribed_values, kappa_state, u0=u0)
        assert result_minus.converged
        R_minus, _, _ = model.residual(result_minus.u, kappa_state)
        loss_minus = R_minus[prescribed_dofs[-1]].item()

        grad_fd[i] = (loss_plus - loss_minus) / (2 * h)

    assign_theta_(model.elem.law, theta0)  # restore

    print("param_names:", param_names)
    print("grad_autograd:", grad_autograd)
    print("grad_fd:", grad_fd)

    denom = grad_fd.abs().clamp_min(1e-8)
    rel_err = (grad_autograd - grad_fd).abs() / denom
    assert torch.all(rel_err < 2e-3), f"relative errors: {rel_err}"
