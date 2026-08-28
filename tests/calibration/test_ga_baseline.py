"""Sanity check: the GA baseline itself converges reasonably (correctness of the GA
code), independent of the cost comparison (which lives in benchmarks/, not asserted in CI since
GA convergence speed is stochastic by nature)."""

import torch

from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.diff import theta_from_law
from diffcohesive.calibration import simulated_response, calibration_loss, genetic_algorithm


def _build_model(T_max_n, T_max_s, G_c1, G_c2, eta, K):
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    ins = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])
    law = BilinearMixedModeTSL(T_max_n=T_max_n, T_max_s=T_max_s, G_c1=G_c1, G_c2=G_c2, eta=eta, K=K)
    return CohesiveMeshModel(
        points=ins.points,
        bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity,
        law=law,
        E=1000.0,
        nu=0.3,
    )


def test_ga_reduces_loss_and_approaches_theta_star():
    torch.manual_seed(0)
    true_vals = dict(T_max_n=6.0, T_max_s=6.0, G_c1=0.08, G_c2=0.08, eta=1.0, K=8000.0)
    true_model = _build_model(**true_vals)

    fixed_dofs = true_model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = true_model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])
    reaction_dof = prescribed_dofs[-1]
    fixed_prefix = torch.zeros(6, dtype=torch.float64)
    disp_values = torch.linspace(0.001, 0.014, 6, dtype=torch.float64).tolist()

    theta_true = theta_from_law(true_model.elem.law)
    P_exp = simulated_response(theta_true, true_model, prescribed_dofs, reaction_dof, disp_values, fixed_prefix)

    guess_model = _build_model(T_max_n=3.0, T_max_s=6.0, G_c1=0.03, G_c2=0.08, eta=1.0, K=4000.0)
    theta_init = theta_from_law(guess_model.elem.law)
    param_names = [name for name, _ in guess_model.elem.law.named_parameters()]
    free_mask = torch.tensor([n in ("T_max_n", "G_c1", "K") for n in param_names])

    def loss_fn(theta):
        return calibration_loss(theta, guess_model, prescribed_dofs, reaction_dof, disp_values, fixed_prefix, P_exp)

    loss_before = loss_fn(theta_init).item()
    result = genetic_algorithm(theta_init, free_mask, loss_fn, pop_size=20, n_generations=40, seed=0)

    assert result.loss < loss_before * 0.2
    assert result.n_evals == 20 * 41
    recovered = dict(zip(param_names, result.theta.tolist()))
    assert abs(recovered["T_max_n"] - true_vals["T_max_n"]) / true_vals["T_max_n"] < 0.3
    assert abs(recovered["G_c1"] - true_vals["G_c1"]) / true_vals["G_c1"] < 0.3
