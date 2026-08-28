"""Quantifies the actual benefit of the sparse bulk-residual conversion (see
assembly/global_assembly.py): wall-clock for a single residual() call's bulk term, dense
(K_bulk @ u) vs. sparse (torch.sparse.mm(K_bulk_sparse, u)), across mesh sizes -- since the two
aren't the same code path any more, this makes the "why bother" concrete rather than asserted.
Run: PYTHONPATH=. python benchmarks/sparse_residual_benchmark.py
"""

import time

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import build_double_cantilever_mesh


def build_and_time(nx, ny, n_repeats=200):
    mesh = build_double_cantilever_mesh(length=15.0, arm_height=1.0, crack_length=5.0, nx=nx, ny=ny)
    law = BilinearMixedModeTSL(T_max_n=5.0, T_max_s=5.0, G_c1=0.05, G_c2=0.05, K=1.0e4)
    model = CohesiveMeshModel(
        points=mesh.points,
        bulk_elements={"triangle": mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity,
        law=law,
        E=1000.0,
        nu=0.3,
    )
    u = torch.randn(model.n_dof, dtype=model.points.dtype)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = model.K_bulk @ u
    t_dense = (time.perf_counter() - t0) / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = torch.sparse.mm(model.K_bulk_sparse, u.unsqueeze(-1)).squeeze(-1)
    t_sparse = (time.perf_counter() - t0) / n_repeats

    return model.n_dof, t_dense, t_sparse


def main():
    print(f"{'n_dof':>8} {'dense (ms)':>12} {'sparse (ms)':>12} {'speedup':>9}")
    for nx, ny in [(10, 2), (30, 4), (60, 8), (100, 10)]:
        n_dof, t_dense, t_sparse = build_and_time(nx, ny)
        print(f"{n_dof:8d} {t_dense * 1000:12.4f} {t_sparse * 1000:12.4f} {t_dense / t_sparse:9.2f}x")


if __name__ == "__main__":
    main()
