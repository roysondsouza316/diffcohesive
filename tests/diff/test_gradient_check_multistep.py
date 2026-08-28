"""Multi-step implicit-diff chaining (the scope extension documented in
diffcohesive/diff/implicit_diff.py): finite-difference
check of dL/dtheta vs. the chained adjoint, where the loss depends on the *last* step's reaction
after a sequence of displacement steps -- so gradients must correctly flow back through every
earlier step's contribution to the current damage/history state, not just the final step's."""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.diff import assign_theta_, solve_diff_path, split_theta, theta_from_law
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.solvers import newton_solve


def _build_model():
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
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


def _step_dofs_values(model, disps):
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])
    dofs_seq, values_seq = [], []
    for d in disps:
        dofs_seq.append(prescribed_dofs)
        values_seq.append(torch.cat([torch.zeros(6, dtype=torch.float64), torch.tensor([d], dtype=torch.float64)]))
    return dofs_seq, values_seq, prescribed_dofs


def _final_reaction_loss(theta, model, dofs_seq, values_seq, kappa0, u0, prescribed_dofs):
    u_list, kappa_list = solve_diff_path(theta, model, dofs_seq, values_seq, kappa0, u0=u0)
    law_params = split_theta(model.elem.law, theta)
    R, _, _ = model.residual(u_list[-1], kappa_list[-2] if len(kappa_list) > 1 else kappa0, law_params=law_params)
    return R[prescribed_dofs[-1]]


def test_multistep_adjoint_gradient_matches_finite_difference():
    model = _build_model()
    disps = [0.004, 0.006, 0.008]  # ramps into and through softening onset over 3 steps
    dofs_seq, values_seq, prescribed_dofs = _step_dofs_values(model, disps)
    kappa0 = model.init_history()
    u0 = torch.zeros(model.n_dof, dtype=torch.float64)

    theta0 = theta_from_law(model.elem.law)
    theta = theta0.clone().requires_grad_(True)

    loss = _final_reaction_loss(theta, model, dofs_seq, values_seq, kappa0, u0, prescribed_dofs)
    loss.backward()
    grad_autograd = theta.grad.clone()
    assert torch.isfinite(grad_autograd).all()

    def forward_value(theta_val):
        assign_theta_(model.elem.law, theta_val)
        u = u0
        kappa = kappa0
        kappa_prev = kappa0
        for dofs, values in zip(dofs_seq, values_seq):
            result = newton_solve(model, dofs, values, kappa, u0=u)
            assert result.converged
            kappa_prev = kappa
            u, kappa = result.u, result.kappa
        # Reaction evaluated against the history state that fed the *last* step's Newton solve,
        # matching solve_diff_path's kappa_list[-2] semantics exactly.
        R, _, _ = model.residual(u, kappa_prev, law_params=None)
        return R[prescribed_dofs[-1]].item()

    grad_fd = torch.zeros_like(theta0)
    for i in range(theta0.numel()):
        h = max(1e-6, 1e-5 * abs(theta0[i].item()))
        theta_plus = theta0.clone()
        theta_plus[i] += h
        loss_plus = forward_value(theta_plus)

        theta_minus = theta0.clone()
        theta_minus[i] -= h
        loss_minus = forward_value(theta_minus)

        grad_fd[i] = (loss_plus - loss_minus) / (2 * h)

    assign_theta_(model.elem.law, theta0)  # restore

    print("grad_autograd:", grad_autograd)
    print("grad_fd:", grad_fd)
    denom = grad_fd.abs().clamp_min(1e-8)
    rel_err = (grad_autograd - grad_fd).abs() / denom
    assert torch.all(rel_err < 5e-3), f"relative errors: {rel_err}"
