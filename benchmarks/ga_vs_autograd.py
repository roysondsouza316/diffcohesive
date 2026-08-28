"""Compare the genetic-algorithm baseline against gradient-based (Adam->L-BFGS) calibration
on the same forward model -- #forward-solves and wall-clock to reach a comparable loss level.
Reporting script, not a pass/fail test (GA convergence speed is stochastic by nature; run with
`python benchmarks/ga_vs_autograd.py`).
"""

import time

import torch

from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.diff import theta_from_law
from diffcohesive.calibration import simulated_response, calibration_loss, calibrate, genetic_algorithm


def build_model(T_max_n, T_max_s, G_c1, G_c2, eta, K):
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    ins = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])
    law = BilinearMixedModeTSL(T_max_n=T_max_n, T_max_s=T_max_s, G_c1=G_c1, G_c2=G_c2, eta=eta, K=K)
    return CohesiveMeshModel(
        points=ins.points, bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity, law=law, E=1000.0, nu=0.3,
    )


def main():
    torch.manual_seed(0)
    true_vals = dict(T_max_n=6.0, T_max_s=6.0, G_c1=0.08, G_c2=0.08, eta=1.0, K=8000.0)
    true_model = build_model(**true_vals)

    fixed_dofs = true_model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = true_model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])
    reaction_dof = prescribed_dofs[-1]
    fixed_prefix = torch.zeros(6, dtype=torch.float64)
    disp_values = torch.linspace(0.001, 0.014, 6, dtype=torch.float64).tolist()

    theta_true = theta_from_law(true_model.elem.law)
    P_exp = simulated_response(theta_true, true_model, prescribed_dofs, reaction_dof, disp_values, fixed_prefix)

    guess_model = build_model(T_max_n=3.0, T_max_s=6.0, G_c1=0.03, G_c2=0.08, eta=1.0, K=4000.0)
    theta_init = theta_from_law(guess_model.elem.law)
    param_names = [name for name, _ in guess_model.elem.law.named_parameters()]
    free_mask = torch.tensor([n in ("T_max_n", "G_c1", "K") for n in param_names])

    def loss_fn(theta):
        return calibration_loss(theta, guess_model, prescribed_dofs, reaction_dof, disp_values, fixed_prefix, P_exp)

    # --- gradient-based (Adam -> L-BFGS) ---
    n_adam, n_lbfgs = 80, 40
    t0 = time.perf_counter()
    theta_grad, loss_grad = calibrate(theta_init, free_mask, loss_fn, n_adam=n_adam, lr_adam=0.08, n_lbfgs=n_lbfgs)
    t_grad = time.perf_counter() - t0
    # Each Adam/L-BFGS step does one forward solve (+ one adjoint backward, no extra forward solves).
    n_evals_grad = n_adam + n_lbfgs  # L-BFGS line search may re-evaluate closure more than once per step

    # --- GA baseline ---
    t0 = time.perf_counter()
    ga_result = genetic_algorithm(theta_init, free_mask, loss_fn, pop_size=24, n_generations=60, seed=0)
    t_ga = time.perf_counter() - t0

    print(f"{'method':<12}{'final loss':>14}{'n_evals':>10}{'wall_clock_s':>14}")
    print(f"{'gradient':<12}{loss_grad:>14.6e}{n_evals_grad:>10}{t_grad:>14.3f}")
    print(f"{'GA':<12}{ga_result.loss:>14.6e}{ga_result.n_evals:>10}{t_ga:>14.3f}")
    print()
    print(f"speedup (n_evals):    {ga_result.n_evals / max(n_evals_grad, 1):.1f}x fewer forward solves for gradient method")
    print(f"speedup (wall-clock): {t_ga / max(t_grad, 1e-9):.1f}x faster wall-clock for gradient method")


if __name__ == "__main__":
    main()
