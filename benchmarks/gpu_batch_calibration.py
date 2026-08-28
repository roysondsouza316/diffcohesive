"""Batched-vs-sequential wall-clock comparison for GPU-batched Newton (over multiple
specimens/test configurations simultaneously): evaluate a batch of candidate
cohesive-law parameter sets (as would happen once per generation in the GA baseline, or once per
population in a batched calibration scheme) either as a sequential Python loop over
``newton_solve`` (mutating the live law each time) or as one ``newton_solve_batched`` call.

Run in the tensormesh-gpu conda env for the GPU comparison:
    PYTHONPATH=. python benchmarks/gpu_batch_calibration.py
(also runs meaningfully on CPU, just without the GPU parallelism payoff).
"""

import time

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.diff import assign_theta_, theta_from_law
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.solvers import newton_solve, newton_solve_batched


def _build_model(device):
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
    return model.to(device)


def run_comparison(batch_size=32, max_iter=30, device="cpu"):
    model = _build_model(device)
    fixed_dofs = model.dof_indices(torch.tensor([0, 1, 2]))
    node3_dofs = model.dof_indices(torch.tensor([3]))
    prescribed_dofs = torch.cat([fixed_dofs, node3_dofs[1:2]])
    d = 0.01
    prescribed_values = torch.cat(
        [torch.zeros(6, dtype=torch.float64, device=device), torch.tensor([d], dtype=torch.float64, device=device)]
    )

    torch.manual_seed(0)
    theta0 = theta_from_law(model.elem.law)
    # A batch of candidate theta's, e.g. as one GA generation's population would look.
    scale = 1.0 + 0.3 * (torch.rand(batch_size, theta0.numel(), dtype=torch.float64, device=device) - 0.5)
    theta_batch = theta0.to(device).unsqueeze(0) * scale

    # -- Sequential: one newton_solve call per candidate, mutating the live law each time. --
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    reactions_seq = []
    kappa0 = model.init_history()
    for i in range(batch_size):
        assign_theta_(model.elem.law, theta_batch[i])
        result = newton_solve(model, prescribed_dofs, prescribed_values, kappa0, max_iter=max_iter)
        reactions_seq.append(result.reaction[-1].item())
    if device == "cuda":
        torch.cuda.synchronize()
    t_seq = time.perf_counter() - t0

    # -- Batched: one newton_solve_batched call across the whole population. --
    param_shapes = [(name, p.shape, p.numel()) for name, p in model.elem.law.named_parameters()]
    law_params_batch = {}
    idx = 0
    for name, shape, numel in param_shapes:
        law_params_batch[name] = theta_batch[:, idx : idx + numel].reshape(batch_size, *shape)
        idx += numel
    kappa0_batch = model.init_history().unsqueeze(0).repeat(batch_size, 1, 1)
    prescribed_values_batch = prescribed_values.unsqueeze(0).repeat(batch_size, 1)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    batched_result = newton_solve_batched(
        model, prescribed_dofs, prescribed_values_batch, law_params_batch, kappa0_batch, max_iter=max_iter
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t_batched = time.perf_counter() - t0

    reactions_batched = batched_result.reaction[:, -1].cpu()
    max_diff = (torch.tensor(reactions_seq) - reactions_batched).abs().max().item()

    print(f"device={device} batch_size={batch_size}")
    print(f"  sequential newton_solve:  {t_seq:.4f} s")
    print(f"  batched newton_solve_batched: {t_batched:.4f} s  (speedup {t_seq / t_batched:.2f}x)")
    print(f"  max |reaction| difference (sequential vs batched): {max_diff:.3e}")


def main():
    run_comparison(batch_size=32, device="cpu")
    if torch.cuda.is_available():
        run_comparison(batch_size=32, device="cuda")
        run_comparison(batch_size=256, device="cuda")


if __name__ == "__main__":
    main()
