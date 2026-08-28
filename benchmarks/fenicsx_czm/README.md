# Cross-code benchmark: FEniCSx interface CZM

Cross-code verification of `diffcohesive` against an independent finite-element code:
the two-inclusion interface-debonding example "Cohesive zone modeling restricted to an
interface" from the COMET-FEniCSx tutorial suite (J. Bleyer,
https://bleyerj.github.io/comet-fenicsx/tours/interfaces/czm_interface_only/czm_interface_only.html),
solved with FEniCSx/DOLFINx 0.9. Both codes use the same gmsh mesh (h = 0.02), the same
Smith--Ferrante cohesive law (G_c = 0.5, sigma_max = 50, beta = 2), and the same boundary
conditions, so the comparison isolates the formulation/solver difference: the reference uses
disconnected submeshes with mixed continuous interpolation and a fixed-point damage iteration;
`diffcohesive` uses zero-thickness cohesive elements from node duplication and full Newton.

Problem: 1.0 x 0.5 plane-strain matrix (E = 3090, nu = 0.25) with two stiff half-disc
inclusions (E = 10000, nu = 0.4, r = 0.25), cohesive behavior on the matrix/inclusion
interfaces only; left edge fixed, right edge pulled to u_x = 0.04.

## Files

- `run_diffcohesive.py` -- the `diffcohesive` model of the problem. Run on Windows/Linux:
  `conda run -n tensormesh python run_diffcohesive.py`
- `run_fenicsx_reference.py` -- runs the published FEniCSx reference solution and writes its
  load-displacement curve to `fenicsx_reference_curve.csv`. It is the tutorial's own script
  with the interactive plotting removed (FEniCSx requires PETSc, so on Windows run it in a
  WSL2 Linux environment with DOLFINx 0.9 installed):
  `python run_fenicsx_reference.py`
- `utils.py` -- the tutorial's helper module (interface-measure utilities), kept under its
  original name because the reference script imports it as `utils`.
- `fenicsx_reference_curve.csv`, `diffcohesive_curve.csv`, `czm_crosscode.png` -- the two
  computed curves and the overlay figure.

Licensing: `run_fenicsx_reference.py` and `utils.py` are derived from the COMET-FEniCSx
tutorials (J. Bleyer) and are redistributed under CC BY-SA 4.0 (see their file headers);
everything else in this repository is under the project's own license.
