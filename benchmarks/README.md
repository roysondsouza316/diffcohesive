# Benchmarks

Performance and cross-code verification scripts. All run from the repository root with
`PYTHONPATH=. python benchmarks/<script>.py` (they are not part of the pytest suite).

- `ga_vs_autograd.py` -- calibration-cost comparison: gradient-based identification
  (Adam -> L-BFGS driven by the adjoint) vs. a derivative-free genetic algorithm on the same
  forward model. Reports forward-solve count, wall-clock time, and final loss for each.
- `gpu_correctness_check.py` -- runs the same cohesive model once on CPU and once on CUDA
  (`CohesiveMeshModel.to('cuda')`) and confirms displacements, damage, and the adjoint
  gradient agree; the gradient is additionally checked against finite differences.
- `gpu_batch_calibration.py` -- wall-clock comparison of evaluating a population of
  cohesive-law parameter sets as a sequential loop vs. the batched Newton solver
  (`torch.func.vmap` over the whole forward solve), on CPU and GPU.
- `sparse_residual_benchmark.py` -- timing of the bulk-residual term computed with the dense
  matrix (`K_bulk @ u`) vs. the sparse path (`torch.sparse.mm`), across mesh sizes.
- `fenicsx_czm/` -- cross-code verification against FEniCSx/DOLFINx on a two-inclusion
  interface-debonding problem (own README inside).
