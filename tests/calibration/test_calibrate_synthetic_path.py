"""Multi-step scope extension of test_calibrate_synthetic.py: recover a synthetic theta* from ONE continuous noisy loading curve (history carried
across displacement steps via solve_diff_path), rather than independent single-step points."""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.calibration import calibrate, calibration_loss_path, simulated_response_path
from diffcohesive.diff import theta_from_law
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import insert_cohesive_interface


def _build_model(T_max_n, T_max_s, G_c1, G_c2, eta, K):
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    ins = insert_cohesive_interface(points, elements, crack_edges=[(0, 2)])
    law = BilinearMixedModeTSL(T_max_n=T_max_n, T_max_s=T_max_s, G_c1=G_c1, G_c2=G_c2, eta=eta, K=K)
    model = CohesiveMeshModel(
        points=ins.points,
        bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity,
        law=law,
        E=1000.0,
        nu=0.3,
    )
    return model


def test_recover_synthetic_theta_star_from_one_continuous_path():
    torch.manual_seed(0)

    true_vals = dict(T_max_n=6.0, T_max_s=6.0, G_c1=0.08, G_c2=0.08, eta=1.0, K=8000.0)
    true_model = _build_model(**true_vals)

    fixed_dofs = true_model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = true_model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])
    reaction_dof = prescribed_dofs[-1]
    fixed_prefix = torch.zeros(6, dtype=torch.float64)

    # One continuous, monotonically increasing loading path (not independent points).
    disp_values = torch.linspace(0.001, 0.014, 6, dtype=torch.float64).tolist()
    theta_true = theta_from_law(true_model.elem.law)
    P_exp = simulated_response_path(theta_true, true_model, prescribed_dofs, reaction_dof, disp_values, fixed_prefix)
    P_exp = P_exp + 0.01 * P_exp.abs().mean() * torch.randn_like(P_exp)

    guess_model = _build_model(T_max_n=3.0, T_max_s=6.0, G_c1=0.03, G_c2=0.08, eta=1.0, K=4000.0)
    theta_init = theta_from_law(guess_model.elem.law)
    param_names = [name for name, _ in guess_model.elem.law.named_parameters()]
    free_mask = torch.tensor([n in ("T_max_n", "G_c1", "K") for n in param_names])

    def loss_fn(theta):
        return calibration_loss_path(
            theta, guess_model, prescribed_dofs, reaction_dof, disp_values, fixed_prefix, P_exp
        )

    loss_before = loss_fn(theta_init).item()
    theta_hat, loss_after = calibrate(theta_init, free_mask, loss_fn, n_adam=80, lr_adam=0.08, n_lbfgs=40)

    assert loss_after < loss_before * 0.05

    recovered = dict(zip(param_names, theta_hat.tolist()))
    assert abs(recovered["T_max_n"] - true_vals["T_max_n"]) / true_vals["T_max_n"] < 0.1
    assert abs(recovered["G_c1"] - true_vals["G_c1"]) / true_vals["G_c1"] < 0.1
    assert abs(recovered["K"] - true_vals["K"]) / true_vals["K"] < 0.15
