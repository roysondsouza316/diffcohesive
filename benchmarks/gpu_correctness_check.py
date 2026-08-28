"""GPU-path correctness check: build the same
cracked 2-triangle model as the existing test suite once on CPU and once on CUDA (via
``CohesiveMeshModel.to('cuda')``), run Newton to the same displacement, and confirm the results
agree -- and separately, that the implicit-diff adjoint gradient still matches finite differences
when everything (model, theta, kappa) lives on GPU. Run in the tensormesh-gpu conda env:

    PYTHONPATH=. python benchmarks/gpu_correctness_check.py
"""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.diff import assign_theta_, solve_diff, split_theta, theta_from_law
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


def run_newton_on(device):
    model = _build_model().to(device)
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])
    prescribed_values = torch.cat(
        [torch.zeros(6, dtype=torch.float64, device=device), torch.tensor([0.01], dtype=torch.float64, device=device)]
    )
    kappa = model.init_history()
    result = newton_solve(model, prescribed_dofs, prescribed_values, kappa)
    return model, prescribed_dofs, result


def check_forward_agreement():
    model_cpu, dofs_cpu, result_cpu = run_newton_on("cpu")
    model_gpu, dofs_gpu, result_gpu = run_newton_on("cuda")

    assert result_cpu.converged and result_gpu.converged
    u_diff = (result_cpu.u - result_gpu.u.cpu()).abs().max().item()
    damage_diff = (result_cpu.damage - result_gpu.damage.cpu()).abs().max().item()
    print(f"[forward] CPU vs CUDA max |u| diff:      {u_diff:.3e}")
    print(f"[forward] CPU vs CUDA max |damage| diff: {damage_diff:.3e}")
    assert u_diff < 1e-8
    assert damage_diff < 1e-8


def check_gpu_gradient_vs_finite_difference():
    device = "cuda"
    model = _build_model().to(device)
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])
    d = 0.008
    prescribed_values = torch.cat(
        [torch.zeros(6, dtype=torch.float64, device=device), torch.tensor([d], dtype=torch.float64, device=device)]
    )
    kappa_state = model.init_history()
    u0 = torch.zeros(model.n_dof, dtype=torch.float64, device=device)

    theta0 = theta_from_law(model.elem.law)
    theta = theta0.clone().requires_grad_(True)

    u_star = solve_diff(theta, model, prescribed_dofs, prescribed_values, kappa_state, u0=u0)
    law_params = split_theta(model.elem.law, theta)
    R, _, _ = model.residual(u_star, kappa_state, law_params=law_params)
    loss = R[prescribed_dofs[-1]]
    loss.backward()
    grad_autograd = theta.grad.clone()

    grad_fd = torch.zeros_like(theta0)
    for i in range(theta0.numel()):
        h = max(1e-6, 1e-5 * abs(theta0[i].item()))
        theta_plus = theta0.clone()
        theta_plus[i] += h
        assign_theta_(model.elem.law, theta_plus)
        result_plus = newton_solve(model, prescribed_dofs, prescribed_values, kappa_state, u0=u0)
        R_plus, _, _ = model.residual(result_plus.u, kappa_state)
        loss_plus = R_plus[prescribed_dofs[-1]].item()

        theta_minus = theta0.clone()
        theta_minus[i] -= h
        assign_theta_(model.elem.law, theta_minus)
        result_minus = newton_solve(model, prescribed_dofs, prescribed_values, kappa_state, u0=u0)
        R_minus, _, _ = model.residual(result_minus.u, kappa_state)
        loss_minus = R_minus[prescribed_dofs[-1]].item()

        grad_fd[i] = (loss_plus - loss_minus) / (2 * h)

    assign_theta_(model.elem.law, theta0)
    rel_err = (grad_autograd.cpu() - grad_fd.cpu()).abs() / grad_fd.cpu().abs().clamp_min(1e-8)
    print(f"[gradient, CUDA] grad_autograd: {grad_autograd.tolist()}")
    print(f"[gradient, CUDA] grad_fd:       {grad_fd.tolist()}")
    print(f"[gradient, CUDA] max relative error: {rel_err.max().item():.3e}")
    assert torch.all(rel_err < 2e-3)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available in this environment -- skipping GPU check.")
        return
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    check_forward_agreement()
    check_gpu_gradient_vs_finite_difference()
    print("All GPU correctness checks passed.")


if __name__ == "__main__":
    main()
