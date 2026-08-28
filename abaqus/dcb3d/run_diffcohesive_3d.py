"""diffcohesive 3D DCB load-displacement curve on the identical mesh/parameters as the
Abaqus COH3D8 reference (generate_inp_3d.py): displacement control with adaptive increment
cutting through the whole softening branch -- the same control method as the Abaqus deck's
*STATIC step, so the two curves are solved like-for-like. (This interface is ductile for the
specimen: the cohesive-zone length E*Gc/sigma0^2 ~ 39 mm spans the ligament, so the softening
branch has no global snap-back and displacement control can trace all of it, in both codes.)

Run from repo root: PYTHONPATH=. python abaqus/dcb3d/run_diffcohesive_3d.py
Writes dcb3d_diffcohesive.csv (delta = tip-line opening, P = total reaction on the top line).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import build_double_cantilever_mesh_3d
from diffcohesive.solvers import adaptive_displacement_solve

HERE = Path(__file__).resolve().parent

L, ARM, W, A0 = 60.0, 2.0, 4.0, 20.0
NX, NY, NZ = 30, 2, 2
E, NU = 70000.0, 0.3
SIGMA0, GC, K, ETA = 30.0, 0.5, 1.0e4, 1.0
# Small viscous damage regularization (the *DAMAGE STABILIZATION analogue) so displacement
# control passes the sharp element pop-ins; kept small relative to the ~30 steps spanning
# the softening branch so the added artificial toughness stays minor.
VISCOSITY = 0.2   # override with argv[1]; argv[2] overrides the output filename

MAX_DISP = 2.5
N_STEPS = 50


def main():
    import sys
    global VISCOSITY
    viscosity = float(sys.argv[1]) if len(sys.argv) > 1 else VISCOSITY
    out_name = sys.argv[2] if len(sys.argv) > 2 else "dcb3d_diffcohesive.csv"
    mesh = build_double_cantilever_mesh_3d(L, ARM, W, A0, nx=NX, ny=NY, nz=NZ)
    law = BilinearMixedModeTSL(T_max_n=SIGMA0, T_max_s=SIGMA0, G_c1=GC, G_c2=GC, eta=ETA, K=K,
                               viscosity=viscosity)
    model = CohesiveMeshModel(
        points=mesh.points, bulk_elements={mesh.cell_type: mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity, law=law, E=E, nu=NU,
    )
    dtype = model.points.dtype

    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    top_y = model.dof_indices(mesh.tip_top_nodes).reshape(-1, 3)[:, 1]
    bot_y = model.dof_indices(mesh.tip_bottom_nodes).reshape(-1, 3)[:, 1]
    n_r, n_t = right_dofs.numel(), top_y.numel()

    def prescribed_fn(d):
        pd = torch.cat([right_dofs, top_y, bot_y])
        pv = torch.cat([
            torch.zeros(n_r, dtype=dtype),
            torch.full((n_t,), d / 2, dtype=dtype),
            torch.full((bot_y.numel(),), -d / 2, dtype=dtype),
        ])
        return pd, pv

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)
    delta_list, P_list = [0.0], [0.0]
    d_prev = 0.0
    for d in torch.linspace(0.0, MAX_DISP, N_STEPS + 1, dtype=dtype)[1:]:
        out = adaptive_displacement_solve(
            model, prescribed_fn, kappa, u, d_prev, float(d),
            initial_step=float(d) - d_prev, max_iter=60,
        )
        if out is None or out[3] < float(d) - 1e-12:
            print(f"stopped at d = {out[3] if out else d_prev:.4f}")
            break
        result, u, kappa, d_prev = out
        P = result.reaction[n_r:n_r + n_t].sum().item()
        delta_list.append(float(d))
        P_list.append(P)
        print(f"d={float(d):.4f}  P={P:.3f}  Dmax={result.damage.max():.3f}")

    out = HERE / out_name
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta", "P"])
        w.writerows(zip(delta_list, P_list))
    print(f"Wrote {out} ({len(delta_list)} points, peak P = {max(P_list):.3f})")


if __name__ == "__main__":
    main()
