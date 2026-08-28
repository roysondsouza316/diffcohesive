"""Lightweight regression check for the mesh/process-zone convergence study:
peak load must decrease (converge downward) as element size drops toward/below the cohesive
zone length, not increase or stay flat. Only two mesh densities, to keep this fast for CI --
see examples/convergence_study.py for the full multi-density sweep and figure."""

from examples.convergence_study import peak_load_for_mesh


def test_peak_load_decreases_with_mesh_refinement():
    peak_coarse, _ = peak_load_for_mesh(nx=8, ny=4)
    peak_fine, _ = peak_load_for_mesh(nx=30, ny=4)
    assert peak_fine < peak_coarse
