# DCB Abaqus reference model

Independent reference solution for `examples/dcb.py`'s mode-I DCB validation, built on the *exact same
mesh* as diffcohesive (generated once in Python, then emitted as plain Abaqus keywords) so the
comparison isolates solver/physics agreement rather than mesh differences.

## Pipeline

1. `python abaqus/dcb/generate_inp.py` (any Python with this repo installed, e.g. the
   `tensormesh` conda env) -- builds the DCB mesh via
   `diffcohesive.mesh.build_double_cantilever_mesh`, using the same parameters as
   `examples/dcb.py`'s `DEFAULT_PARAMS`, and writes `dcb.inp`: CPE3 plane-strain triangles for
   the arms, COH2D4 cohesive elements for the bonded ligament with a BK mixed-mode,
   energy-based, linear-softening traction-separation response mapped directly from
   `BilinearMixedModeTSL`'s `(T_max_n, T_max_s, G_c1, G_c2, eta, K)`.
2. `& "C:\SIMULIA\Commands\abaqus.bat" job=dcb_ref input=dcb.inp interactive` -- solves with
   Abaqus/Standard (*STATIC, automatic incrementation, `NLGEOM=NO` to match the bulk law's
   small-strain assumption).
3. `& "C:\SIMULIA\Commands\abaqus.bat" python extract_odb.py` -- **must** use Abaqus's own
   embedded Python (`odbAccess` is not installable elsewhere), not this repo's conda env. Writes
   `dcb_abaqus_result.csv` (opening displacement vs. reaction load history at the two tip nodes).
4. `python abaqus/dcb/compare_abaqus.py` (back in a normal Python env) -- runs
   `examples/dcb.py`'s own solver, overlays it against the Abaqus CSV and the LEFM beam-theory
   line, and reports the peak-load agreement.

## Result (default parameters)

Peak loads agree to ~5%: diffcohesive 0.4305 at delta=0.498 vs. Abaqus 0.4088 at delta~0.51; both
match the pre-peak LEFM compliance line closely. See `dcb_comparison.png`.

## Regenerating

`dcb.inp`, `dcb_abaqus_result.csv`, and `dcb_comparison.png` are committed for reference/
reproducibility. The heavier Abaqus job artifacts (`.odb`, `.dat`, `.msg`, `.sta`, `.com`,
`.prt`) are gitignored -- rerun steps 2-3 above to regenerate them.
