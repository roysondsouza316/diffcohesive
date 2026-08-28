"""diffcohesive reproduction of the FEniCSx interface-CZM example "Cohesive zone modeling restricted to
an interface" (czm_interface_only): 1.0 x 0.5 plane-strain matrix with two half-disc
inclusions (r = 0.25, centers (0.4, 0) and (0.6, 0.5)), Smith-Ferrante cohesive law
(G_c = 0.5, sigma_max = 50, beta = 2) on the matrix/inclusion interfaces only, left edge
fixed, right edge pulled to u_x = 0.04 (u_y = 0) in 40 increments.

The FEniCSx reference (run via run_fenicsx_reference.py, DOLFINx 0.9.0)
uses disconnected submeshes + mixed CG interpolation and a fixed-point damage iteration; the
present model uses zero-thickness cohesive elements from node duplication + full Newton on the
smoothed law. Same mesh generator (gmsh, h = 0.02), same constitutive parameters, same BCs --
the comparison isolates the formulation/solver difference. Known formulation difference,
stated: the reference's effective opening uses the raw normal jump; ours Macaulay-brackets it
(compression fully stiff), which is inactive in this tension-dominated debonding problem.

Run: PYTHONPATH=../../ conda run -n tensormesh python run_diffcohesive.py
Outputs diffcohesive_curve.csv and (if the reference CSV exists) czm_crosscode.png.
"""

import csv
import math
from pathlib import Path

import gmsh
import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diffcohesive.assembly import CohesiveMeshModel, assemble_cst_stiffness, orthotropic_plane_C
from diffcohesive.laws import SmithFerranteTSL
from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.solvers import adaptive_displacement_solve

HERE = Path(__file__).resolve().parent

L, H, R = 1.0, 0.5, 0.25
C1, C2 = (0.4, 0.0), (0.6, 0.5)
E_MAT, NU_MAT = 3090.0, 0.25
E_INC, NU_INC = 10000.0, 0.4
GC, SIGMA_C, BETA = 0.5, 50.0, 2.0
H_MESH = 0.02
U_MAX, N_INCR = 0.04, 40


def build_mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, L, H)
    d1 = occ.addDisk(*C1, 0, R, R)
    d2 = occ.addDisk(*C2, 0, R, R)
    occ.fragment([(2, rect)], [(2, d1), (2, d2)])
    occ.synchronize()
    # Discard fragments outside the rectangle (the halves of the discs beyond the edges).
    for dim, tag in gmsh.model.getEntities(2):
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if not (-1e-9 <= x <= L + 1e-9 and -1e-9 <= y <= H + 1e-9):
            gmsh.model.occ.remove([(dim, tag)], recursive=False)
    occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", H_MESH)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", H_MESH)
    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(node_tags)
    coords = coords.reshape(-1, 3)[order][:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(np.array(node_tags)[order])}

    tris = []
    for dim, tag in gmsh.model.getEntities(2):
        etypes, _, enodes = gmsh.model.mesh.getElements(dim, tag)
        for et, nn in zip(etypes, enodes):
            if et == 2:  # 3-node triangle
                tris.append(np.array([tag_to_idx[int(t)] for t in nn]).reshape(-1, 3))
    tris = np.vstack(tris)
    gmsh.finalize()

    # Compact away nodes not referenced by any kept triangle (gmsh keeps nodes of the removed
    # outside-of-rectangle fragments and of boundary curves; unreferenced nodes would leave
    # zero rows in the stiffness -> singular solve).
    used = np.unique(tris)
    remap = -np.ones(coords.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    coords = coords[used]
    tris = remap[tris]
    return torch.tensor(coords, dtype=torch.float64), torch.tensor(tris, dtype=torch.long)


def classify(points, tris):
    cent = points[tris].mean(dim=1)
    def inside(c):
        return ((cent[:, 0] - c[0]) ** 2 + (cent[:, 1] - c[1]) ** 2) < R ** 2 - 1e-10
    return inside(C1) | inside(C2)  # True = inclusion


def interface_edges(tris, is_inc):
    edge_owner = {}
    for ei, row in enumerate(tris.tolist()):
        for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0])):
            edge_owner.setdefault((min(a, b), max(a, b)), []).append(ei)
    edges = []
    for (a, b), owners in edge_owner.items():
        if len(owners) == 2 and bool(is_inc[owners[0]]) != bool(is_inc[owners[1]]):
            edges.append((a, b))
    return edges


def main():
    points, tris = build_mesh()
    is_inc = classify(points, tris)
    edges = interface_edges(tris, is_inc)
    print(f"mesh: {points.shape[0]} nodes, {tris.shape[0]} triangles, "
          f"{int(is_inc.sum())} inclusion tris, {len(edges)} interface edges")

    ins = insert_cohesive_interface(points, tris, crack_edges=edges)

    C_mat = orthotropic_plane_C(E1=E_MAT, E2=E_MAT, nu12=NU_MAT,
                                 G12=E_MAT / (2 * (1 + NU_MAT)), E3=E_MAT, state="plane_strain")
    C_inc = orthotropic_plane_C(E1=E_INC, E2=E_INC, nu12=NU_INC,
                                 G12=E_INC / (2 * (1 + NU_INC)), E3=E_INC, state="plane_strain")
    K_bulk = (assemble_cst_stiffness(ins.points, ins.elements[~is_inc], C_mat)
              + assemble_cst_stiffness(ins.points, ins.elements[is_inc], C_inc))

    law = SmithFerranteTSL(Gc=GC, sigma_c=SIGMA_C, beta=BETA)
    model = CohesiveMeshModel(
        points=ins.points, bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=ins.cohesive_connectivity, law=law, bulk_stiffness=K_bulk,
    )
    dtype = model.points.dtype
    p = model.points
    tol = 1e-9
    left = torch.nonzero(p[:, 0].abs() < tol, as_tuple=True)[0]
    right = torch.nonzero((p[:, 0] - L).abs() < tol, as_tuple=True)[0]
    left_dofs = model.dof_indices(left)
    right_dofs = model.dof_indices(right).reshape(-1, 2)
    right_x, right_y = right_dofs[:, 0], right_dofs[:, 1]

    def prescribed_fn(t):
        pd = torch.cat([left_dofs, right_x, right_y])
        pv = torch.cat([torch.zeros(left_dofs.numel(), dtype=dtype),
                        torch.full((right_x.numel(),), t, dtype=dtype),
                        torch.zeros(right_y.numel(), dtype=dtype)])
        return pd, pv

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)
    ts, Fs = [0.0], [0.0]
    t_prev = 0.0
    for t in np.linspace(0.0, U_MAX, N_INCR + 1)[1:]:
        out = adaptive_displacement_solve(
            model, prescribed_fn, kappa, u, t_prev, float(t),
            initial_step=float(t) - t_prev, max_iter=80,
        )
        if out is None or out[3] < t - 1e-12:
            print(f"stopped at u_x = {out[3] if out else t_prev:.5f}")
            break
        result, u, kappa, t_prev = out
        n_l = left_dofs.numel()
        F = result.reaction[n_l:n_l + right_x.numel()].sum().item()
        ts.append(float(t))
        Fs.append(F)
        print(f"u={t:.5f}  F={F:.3f}  Dmax={result.damage.max():.3f}")

    with open(HERE / "diffcohesive_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta", "F"])
        w.writerows(zip(ts, Fs))
    print("wrote diffcohesive_curve.csv, peak F =", max(Fs))

    ref = HERE / "fenicsx_reference_curve.csv"
    if ref.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rd, rf = [], []
        with open(ref, newline="") as f:
            for row in csv.DictReader(f):
                rd.append(float(row["delta"]))
                rf.append(float(row["F"]))
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(rd, rf, "s--", ms=4, color="C1",
                  label="FEniCSx/DOLFINx 0.9 (reference tour, submesh CG)")
        plt.plot(ts, Fs, "o-", ms=3, color="C0",
                  label="present (diffcohesive, cohesive elements)")
        plt.xlabel("imposed displacement $u_x$")
        plt.ylabel("reaction force $F_x$")
        plt.title("Two-inclusion interface debonding: cross-code comparison", fontsize=10)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(HERE / "czm_crosscode.png", dpi=160)
        print("wrote czm_crosscode.png")


if __name__ == "__main__":
    main()
