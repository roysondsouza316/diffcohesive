"""Implicit-diff adjoint gradient check for the FRICTIONAL cohesive law (closing the
documented gap: the friction law was forward-validated only). Same finite-difference-vs-adjoint
protocol as the bilinear-law checks, on the compressed+sheared two-quad sandwich, with the
interface partially damaged and actively slipping so every ingredient of the law -- damage,
unilateral contact, and the smoothed Coulomb return map -- contributes to the gradient. All
seven parameters (sigma0, tau0, G_c1, G_c2, K1, K2, mu) are checked."""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.diff import assign_theta_, solve_diff, split_theta, theta_from_law
from diffcohesive.laws import FrictionalCohesiveTSL
from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.solvers import newton_solve


def _build_model():
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 3, 2], [2, 3, 5, 4]], dtype=torch.long)
    ins = insert_cohesive_interface(points, elements, crack_edges=[(2, 3)])
    law = FrictionalCohesiveTSL(
        sigma0=3.0, tau0=3.0, G_c1=0.1, G_c2=0.1, K1=1.0e4, K2=1.0e4, mu=0.5
    )
    model = CohesiveMeshModel(
        points=ins.points,
        bulk_elements={"quad": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity,
        law=law,
        E=1.0e6,
        nu=0.0,
    )
    return model, ins


def _prescribed(model, ins, shear, compress):
    dtype = torch.float64
    bottom_dofs = model.dof_indices(torch.tensor([0, 1]))
    top_nodes = torch.unique(ins.elements[1])
    top_dofs = model.dof_indices(top_nodes).reshape(-1, 2)
    top_x, top_y = top_dofs[:, 0], top_dofs[:, 1]
    prescribed_dofs = torch.cat([bottom_dofs, top_x, top_y])
    prescribed_values = torch.cat(
        [
            torch.zeros(bottom_dofs.numel(), dtype=dtype),
            torch.full((top_x.numel(),), shear, dtype=dtype),
            torch.full((top_y.numel(),), compress, dtype=dtype),
        ]
    )
    return prescribed_dofs, prescribed_values, top_x


def test_friction_adjoint_gradient_matches_finite_difference():
    model, ins = _build_model()
    sc2 = 2.0 * 0.1 / 3.0
    compress = -1.0e-3
    # Precondition the history into the partially-damaged, actively-slipping regime, then
    # gradient-check one further step from that (fixed) state.
    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=torch.float64)
    for s in torch.linspace(0.0, 0.55 * sc2, 12, dtype=torch.float64):
        pd, pv, _ = _prescribed(model, ins, s.item(), compress)
        result = newton_solve(model, pd, pv, kappa, u0=u, max_iter=80)
        assert result.converged
        u, kappa = result.u, result.kappa
    assert 0.05 < result.damage.max().item() < 0.999  # partially damaged
    assert kappa[..., 1].abs().max().item() > 0.0     # slip has accumulated

    shear_next = 0.62 * sc2
    pd, pv, top_x = _prescribed(model, ins, shear_next, compress)
    kappa_state = kappa.detach()
    u0 = u.detach()

    theta0 = theta_from_law(model.elem.law)
    theta = theta0.clone().requires_grad_(True)

    def loss_from(theta_vec):
        u_star = solve_diff(theta_vec, model, pd, pv, kappa_state, u0=u0)
        law_params = split_theta(model.elem.law, theta_vec)
        R, _, _ = model.residual(u_star, kappa_state, law_params=law_params)
        return R[top_x].sum()  # shear reaction transmitted across the interface

    loss = loss_from(theta)
    loss.backward()
    grad_autograd = theta.grad.clone()
    assert torch.isfinite(grad_autograd).all()

    param_names = [name for name, _ in model.elem.law.named_parameters()]
    grad_fd = torch.zeros_like(theta0)
    for i in range(theta0.numel()):
        h = max(1e-7, 1e-6 * abs(theta0[i].item()))
        vals = []
        for sign in (+1.0, -1.0):
            theta_pert = theta0.clone()
            theta_pert[i] += sign * h
            assign_theta_(model.elem.law, theta_pert)
            res = newton_solve(model, pd, pv, kappa_state, u0=u0, max_iter=80)
            assert res.converged
            R, _, _ = model.residual(res.u, kappa_state)
            vals.append(R[top_x].sum().item())
        grad_fd[i] = (vals[0] - vals[1]) / (2 * h)
    assign_theta_(model.elem.law, theta0)

    print("params:", param_names)
    print("adjoint:", grad_autograd.tolist())
    print("fd:     ", grad_fd.tolist())
    denom = grad_fd.abs().clamp_min(1e-6 * grad_fd.abs().max())
    rel_err = (grad_autograd - grad_fd).abs() / denom
    assert torch.all(rel_err < 5e-3), f"relative errors: {rel_err}"
    # mu must carry a genuinely nonzero sensitivity in this slipping state.
    mu_idx = param_names.index("mu")
    assert grad_fd[mu_idx].abs().item() > 1e-6