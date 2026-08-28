# Examples

## After `pip install diffcohesive` (no repository needed)

Start here. This script imports only the installed package and runs from any directory:

- `quickstart_dcb.py` -- a differentiable double-cantilever beam in ~20 lines: build the
  model, solve, and get gradients of the response with respect to the cohesive-law
  parameters. `python quickstart_dcb.py`

The same pattern extends directly: pick a law from `diffcohesive.laws` (bilinear, the
Alfano shape library, friction, Smith-Ferrante, or the neural-network law), a mesh from
`diffcohesive.mesh` (or your own `points`/`elements` arrays plus the edges to place cohesive
elements on), and a solver from `diffcohesive.solvers` (Newton, arc-length, or the
GPU-batched variant). Calibration against measured data goes through
`diffcohesive.calibration` -- see the README's calibration section.

## Validation studies (run from a repository clone)

These reproduce the package's published validations; they write figures/CSVs into the
repository and are run as `PYTHONPATH=. python examples/<script>.py` from the repository
root:

- `dcb.py` -- mode-I DCB against LEFM beam theory (and the Abaqus reference in `abaqus/`).
- `mixed_mode_beam.py` -- mesh-level mixed-mode dissipation against the Benzeggagh-Kenane
  criterion across loading angles.
- `alfano_shape_study.py` -- the four-benchmark interface-law shape study (bilinear,
  linear-parabolic, exponential, trapezoidal envelopes).
- `crack_propagation.py` -- process-zone formation and crack advance against the LEFM
  prediction.
- `crack_path_selection.py` -- competition between candidate crack paths.
- `friction_demo.py` -- the Alfano-Sacco damage-friction law under normal compression.
- `convergence_study.py` -- mesh/process-zone convergence of the DCB peak load.
