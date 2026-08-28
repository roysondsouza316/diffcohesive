"""GPU-batched Newton over multiple specimens/test configurations:
newton_solve_batched must agree with a sequential loop of newton_solve, one call per candidate
theta, to near machine precision -- both are solving the exact same fixed-history equilibrium
problem, just batched vs. looped. (An earlier implementation incorrectly threaded the history
variable forward every Newton *iteration* instead of holding it fixed for the whole load step,
which this test would have caught: it showed up as a small but very real few-tenths-of-a-percent
reaction-force mismatch that got *worse*, not better, with more iterations.)"""

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.diff import assign_theta_, theta_from_law
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.solvers import newton_solve, newton_solve_batched


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


def test_batched_matches_sequential_across_a_parameter_population():
    torch.manual_seed(0)
    model = _build_model()
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])
    d = 0.006
    prescribed_values = torch.cat([torch.zeros(6, dtype=torch.float64), torch.tensor([d], dtype=torch.float64)])

    batch_size = 8
    theta0 = theta_from_law(model.elem.law)
    scale = 1.0 + 0.3 * (torch.rand(batch_size, theta0.numel(), dtype=torch.float64) - 0.5)
    theta_batch = theta0.unsqueeze(0) * scale

    kappa0 = model.init_history()
    reactions_seq = []
    for i in range(batch_size):
        assign_theta_(model.elem.law, theta_batch[i])
        result = newton_solve(model, prescribed_dofs, prescribed_values, kappa0, max_iter=50, tol=1e-12)
        assert result.converged
        reactions_seq.append(result.reaction[-1].item())
    reactions_seq = torch.tensor(reactions_seq, dtype=torch.float64)

    param_shapes = [(name, p.shape, p.numel()) for name, p in model.elem.law.named_parameters()]
    law_params_batch, idx = {}, 0
    for name, shape, numel in param_shapes:
        law_params_batch[name] = theta_batch[:, idx : idx + numel].reshape(batch_size, *shape)
        idx += numel
    kappa0_batch = kappa0.unsqueeze(0).repeat(batch_size, 1, 1)
    prescribed_values_batch = prescribed_values.unsqueeze(0).repeat(batch_size, 1)

    batched = newton_solve_batched(
        model, prescribed_dofs, prescribed_values_batch, law_params_batch, kappa0_batch, max_iter=30
    )
    reactions_batched = batched.reaction[:, -1]

    max_diff = (reactions_seq - reactions_batched).abs().max().item()
    assert max_diff < 1e-5
