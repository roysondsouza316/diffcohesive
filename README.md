# diffcohesive

A GPU-accelerated, end-to-end differentiable cohesive-zone fracture solver for gradient-based
identification of analytic and neural-network-based traction-separation laws. Built on
[TensorMesh](https://www.tensor-mesh.com/).

## What this is (packaging / architecture)

`diffcohesive` is a **standalone, pip-installable Python package** (a library with its own API),
not a TensorMesh plugin or fork. TensorMesh is a dependency (`tensormesh-fem` on PyPI), used for
one thing: assembling the bulk-elasticity stiffness from its native element library. Everything
cohesive -- interface insertion, the zero-thickness cohesive element, the traction-separation
laws, the nonlinear solvers (Newton + Crisfield arc-length with increment cutting), and the
implicit-differentiation adjoint -- lives in this package. TensorMesh has no plugin mechanism and
its assembly is hard-wired to one-DOF-set-per-mesh-node, which is why cohesive elements are
realized by mesh-level node duplication feeding a custom assembly rather than injected into
TensorMesh itself.

```bash
pip install diffcohesive   # released on PyPI

# or, for development, from a clone:
pip install -e ".[dev]"    # editable install + pytest
python -m pytest tests/    # full validation suite (75 tests)
```

**Installing as a top-up to an existing TensorMesh environment (the normal case).** Run
`pip install diffcohesive` *inside the environment where `tensormesh-fem` and your (CUDA)
PyTorch already live* — pip then reports every dependency as already satisfied and adds only
the ~50 kB `diffcohesive` package itself. Run in a bare environment instead, pip must build the
whole stack from scratch, and on Windows/PyPI that means the **CPU-only** torch wheel: if you
want GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/)
first, then `pip install diffcohesive` on top. A wheel never bundles its dependencies —
`tensormesh-fem` *requiring* torch does not mean it *contains* torch; pip installs each package
of the tree separately, once per environment.

### Supported TensorMesh versions and upgrading

`diffcohesive` touches only TensorMesh's stable core (`Mesh`,
`LinearElasticityElementAssembler`, sparse-matrix export), so TensorMesh's 0.2.0 additions
(mixed multi-field assembly, distributed FEM, wave operators) neither affect nor constrain it.
Verified combinations (full test suite + CPU-vs-CUDA end-to-end solve/adjoint comparison at
machine precision + functional GPU sparse-solve checks):

| package            | verified (old)  | verified (current) | constraint source                     |
|--------------------|-----------------|--------------------|---------------------------------------|
| `tensormesh-fem`   | 0.1.1           | **0.2.0**          | this package: `>=0.1.1`               |
| `torch-sla`        | 0.2.1           | **0.3.2**          | tensormesh-fem 0.2.0 needs `>=0.3.0`  |
| `torch`            | 2.11.0+cu128    | 2.11.0+cu128       | tensormesh-fem: `>=2.0`               |
| `cupy-cuda12x`     | 14.1.1          | 14.1.1             | must match torch's CUDA line (cu12)   |
| `nvmath-python`    | 0.9.0 `[cu12]`  | 0.9.0 `[cu12]`     | torch-sla `[cudss]` extra             |
| `nvidia-cudss-cu12`| 0.7.1.6         | 0.7.1.6            | pulled by nvmath `[cu12]`             |

The one rule for the GPU stack: **keep everything on the same CUDA major line as your torch
build** (`+cu128` → `cupy-cuda12x`, `nvmath-python[cu12]`, `nvidia-cudss-cu12`). No extra can
install the CUDA build of *torch itself* — get that from PyTorch's own selector first; the
extras then match around it. To upgrade an existing CUDA-12 environment, one command:

```bash
pip install -U "tensormesh-fem[cudss]"   # Windows and Linux: TM 0.2 + torch-sla 0.3 + cuDSS chain
```

(On Linux, `tensormesh-fem[gpu]` also works and adds PyAMG. Do **not** use `[gpu]` on Windows:
it requires `torch-sla[all]`, whose `torch-amgx` backend ships no Windows wheels, so pip's
resolver fails.) `diffcohesive`'s GPU execution (`model.to('cuda')`) uses `torch.linalg`
directly and needs none of the extras; torch-sla's `cudss`/CuPy backends activate automatically
when the cu12 libraries in the table are present (`torch_sla.show_backends()` to confirm;
verified here: cuDSS direct solve on an assembled stiffness agrees with a dense reference to
~5e-15).

## Quickstart: a differentiable DCB in ~20 lines

```python
import torch
from diffcohesive.mesh import build_double_cantilever_mesh
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.solvers import newton_solve
from diffcohesive.diff import solve_diff, theta_from_law

# 1. FE model: DCB with a 5 mm pre-crack, cohesive elements on the bonded ligament
mesh = build_double_cantilever_mesh(length=15.0, arm_height=1.0,
                                    crack_length=5.0, nx=30, ny=4)
law = BilinearMixedModeTSL(T_max_n=5.0, T_max_s=5.0, G_c1=0.05, G_c2=0.05, K=1e4)
model = CohesiveMeshModel(mesh.points, {mesh.cell_type: mesh.elements},
                          mesh.cohesive_connectivity, law, E=1000.0, nu=0.3)

# 2. boundary conditions: clamp the far edge, open the arm tips by +/- d/2
right = model.dof_indices(mesh.right_edge_nodes)
tip_t = model.dof_indices(torch.tensor([mesh.tip_top]))[1]
tip_b = model.dof_indices(torch.tensor([mesh.tip_bottom]))[1]
d = 0.3
dofs = torch.cat([right, tip_t.reshape(1), tip_b.reshape(1)])
vals = torch.cat([torch.zeros(right.numel(), dtype=torch.float64),
                  torch.tensor([d/2, -d/2], dtype=torch.float64)])

# 3a. plain forward solve...
result = newton_solve(model, dofs, vals, model.init_history())
print("reaction:", result.reaction[-2].item(), "max damage:", result.damage.max().item())

# 3b. ...or a DIFFERENTIABLE solve: d(anything)/d(law parameters) via the adjoint
theta = theta_from_law(law).requires_grad_(True)
u_star = solve_diff(theta, model, dofs, vals, model.init_history())
u_star.norm().backward()
print("d||u||/d(T_max, G_c, ...):", theta.grad)
```

This quickstart ships as a runnable script: `examples/quickstart_dcb.py` needs only the
installed package (`python quickstart_dcb.py` from anywhere). `examples/README.md` separates
what runs pip-only from the validation studies that are run from a repository clone.

3D is the same API with `build_double_cantilever_mesh_3d` (hexahedra + 8-node cohesive faces);
orthotropic bulk plugs in via `bulk_stiffness=assemble_cst_stiffness(pts, tris,
orthotropic_plane_C(E1=..., E2=..., nu12=..., G12=...))`. Every validation figure is
regenerated by a committed script under `examples/` or `benchmarks/`.

## Supported elements and bulk constitutive models

| Role | Elements | Notes |
|---|---|---|
| Bulk 2D (via TensorMesh) | 3-node triangles, 4-node quadrilaterals | plane strain, isotropic (E, nu) |
| Bulk 2D orthotropic (this package) | 3-node triangles | `assembly/orthotropic.py`: plane strain/stress from (E1, E2, nu12, G12) + material angle; isotropic limit matches TensorMesh entry-wise |
| Bulk 3D (via TensorMesh) | 8-node hexahedra | isotropic |
| Cohesive 2D (this package) | 4-node zero-thickness (COH2D4-equivalent) | 2-point Gauss, autograd tangent |
| Cohesive 3D (this package) | 8-node zero-thickness (COH3D8-equivalent) | 2x2 Gauss, autograd tangent; same law implementations as 2D |

Interface insertion: edge-based in 2D (`insert_cohesive_interface`, any cyclically-ordered
element type) and face-based in 3D (`insert_cohesive_interface_3d`, hexahedra), both with
connected-components node duplication, partial bonding (pre-cracks), and opening-positive
orientation. Multiple candidate paths may be inserted simultaneously -- the mechanics then
selects the failing path (`examples/crack_path_selection.py`).

## Traction-separation laws

- `BilinearMixedModeTSL` -- Camanho-Davila/Alfano-Crisfield bilinear with Turon mixed-mode
  onset/final-displacement closure and a choice of mixed-mode fracture criterion:
  **Benzeggagh-Kenane** (`mixed_mode_criterion="bk"`) or the **power law**
  (`"power"`, (G_I/G_Ic)^a + (G_II/G_IIc)^a = 1). Fully differentiable / calibratable.
- Shape library (`laws/shapes.py`, after Alfano CST 66 (2006) 723-730): **bilinear,
  linear-parabolic, exponential, trapezoidal** envelopes from identical (K0, sigma0, Gc), with
  closed-form internal parameters verified against the paper's own computed values.
- `FrictionalCohesiveTSL` (`laws/friction.py`, after Alfano & Sacco IJNME 68 (2006) 542-582):
  interface damage **combined with Coulomb friction** acting on the damaged area fraction --
  residual shear = mu*|contact stress| after full decohesion, dissipation = G_c + friction work.
- `NeuralMonotoneTSL` -- monotone neural damage law (architecturally admissible: D in [0,1],
  dD/dkappa >= 0), calibratable by the same gradient loop.

## Validation (all regenerable from `examples/` and `tests/`)

- **Mode I DCB** vs LEFM beam theory (~2% pre-peak compliance) and an independent
  **Abaqus COH2D4 reference model on the identical mesh** (~5% peak load) -- `abaqus/dcb/`.
- **Alfano CST 2006 benchmark suite** (`examples/alfano_shape_study.py`): thin DCB
  (shape-insensitivity), thick DCB and rigid compact specimen (peak-load ordering trapezoidal >
  linear-parabolic > bilinear ~ exponential, as in the paper's Figs 6/8), and the **mode II
  pull-out test** (load-displacement curve inheriting the interface-law shape, paper's Fig 10).
- **Mixed mode at the mesh level**: dissipated energy matches the BK prediction to <0.6% across
  the full mode-mix sweep (`examples/mixed_mode_beam.py`); power-law criterion validated at the
  point level.
- **Friction**: constitutive and mesh-level Coulomb checks (`tests/laws/test_friction.py`,
  `tests/assembly/test_friction_mesh.py`).
- **Crack initiation & propagation** made explicit (`examples/crack_propagation.py`): damage
  front vs LEFM crack-length prediction, process-zone profiles, deformed meshes.
- **Mesh/process-zone convergence** (`examples/convergence_study.py`).
- **Gradients**: implicit-diff adjoint vs finite differences on every parameter, single-step and
  multi-step (path-dependent history) -- the package's core claim (`tests/diff/`).
- **GPU**: CUDA forward/adjoint correctness + vmap-batched Newton over parameter populations
  (7-55x wall-clock, `benchmarks/`).

## Layout

- `diffcohesive/laws/` -- traction-separation laws (bilinear+BK/power, shape library, friction,
  neural), smoothed primitives.
- `diffcohesive/mesh/` -- cohesive node-duplication insertion (tri/quad), DCB and pull-out
  specimen builders.
- `diffcohesive/elements/` -- cohesive element residual/tangent (autograd).
- `diffcohesive/assembly/`, `diffcohesive/solvers/`, `diffcohesive/diff/` -- global assembly
  (vmap-vectorized), nonlinear solvers (Newton with increment cutting, adaptive Crisfield
  arc-length, batched Newton), implicit-diff adjoint (single and multi-step).
- `diffcohesive/calibration/` -- losses (single-step and path-consistent), Adam->L-BFGS,
  GA baseline.
- `examples/`, `benchmarks/`, `tests/` -- validation scripts (DCB, mixed-mode, shape study,
  crack propagation), performance/cross-code benchmarks, test suite.
- `abaqus/` -- independent Abaqus COH2D4 reference models (input-deck generators, extraction
  and comparison scripts) used for the cross-check validations.

Known limitations, stated openly: 3D is validated against beam theory and element-level gates
but not yet against a 3D Abaqus/literature benchmark; orthotropic bulk is 2D/CST only; no
literal ENF/MMB contact fixtures (no contact/no-penetration formulation beyond the interface's
own compression penalty); crack paths are limited to supplied candidate interfaces.
