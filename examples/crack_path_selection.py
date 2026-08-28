"""Crack-path selection among multiple candidate interfaces. In an intrinsic CZM the path is
chosen by the mechanics among the candidate interfaces provided, not prescribed in advance:
provide one candidate and the path is fixed; provide several and the energetically favorable
one localizes while the others unload.

Demonstration: a single-edge-notched plate in tension with TWO fully-independent candidate
paths of the SAME cohesive material: path A continues straight ahead of the notch (mid-height);
path B is an un-notched plane at 3/4 height. Both are given cohesive elements; under tension
the stress concentration at the notch tip localizes damage on path A, which propagates to
complete failure, while path B's damage stays low everywhere and unloads elastically.

Produces examples/crack_path_selection.png. Run:
    PYTHONPATH=. python examples/crack_path_selection.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import insert_cohesive_interface
from diffcohesive.solvers import adaptive_displacement_solve

W, H = 20.0, 20.0
NXC, NYC = 40, 40           # grid cells
NOTCH = 6.0                  # traction-free notch length on path A (from the left edge)
E, NU = 70000.0, 0.3
# G_c chosen so the process zone is long relative to the ligament (l_cz = E*Gc/sigma0^2 ~ 31 mm
# > W): crack growth is then stable under displacement control and no arc-length continuation
# is needed -- the demo stays on the physical branch by construction.
SIGMA0, GC, K = 30.0, 0.4, 1.0e5


def build():
    n_cols, n_rows = NXC + 1, NYC + 1
    dx, dy = W / NXC, H / NYC

    def gid(ix, iy):
        return iy * n_cols + ix

    pts = torch.tensor([[ix * dx, iy * dy] for iy in range(n_rows) for ix in range(n_cols)],
                        dtype=torch.float64)
    tris = []
    for iy in range(NYC):
        for ix in range(NXC):
            p00, p10 = gid(ix, iy), gid(ix + 1, iy)
            p01, p11 = gid(ix, iy + 1), gid(ix + 1, iy + 1)
            tris.append([p00, p10, p11])
            tris.append([p00, p11, p01])
    tris = torch.tensor(tris, dtype=torch.long)

    iy_A, iy_B = NYC // 2, (3 * NYC) // 4
    path_A = [(gid(ix, iy_A), gid(ix + 1, iy_A)) for ix in range(NXC)]
    path_B = [(gid(ix, iy_B), gid(ix + 1, iy_B)) for ix in range(NXC)]
    ins = insert_cohesive_interface(pts, tris, crack_edges=path_A + path_B)

    # Bond everything except the notch cells of path A (x < NOTCH at mid-height).
    mid_y_A, mid_y_B = iy_A * dy, iy_B * dy
    coh = ins.cohesive_connectivity
    coh_mid = ins.points[coh[:, :2]].mean(dim=1)
    on_A = (coh_mid[:, 1] - mid_y_A).abs() < 1e-9
    notch = on_A & (coh_mid[:, 0] < NOTCH - 1e-9)
    bonded = ~notch
    return ins, coh[bonded], coh_mid[bonded], on_A[bonded], pts.shape[0], dy


def main():
    ins, coh, coh_mid, on_A, _, dy = build()
    law = BilinearMixedModeTSL(T_max_n=SIGMA0, T_max_s=SIGMA0, G_c1=GC, G_c2=GC, eta=1.0, K=K)
    model = CohesiveMeshModel(
        points=ins.points,
        bulk_elements={"triangle": ins.elements},
        cohesive_connectivity=coh,
        law=law,
        E=E,
        nu=NU,
    )
    dtype = model.points.dtype
    p = model.points
    tol = 1e-9
    bottom_nodes = torch.nonzero(p[:, 1].abs() < tol, as_tuple=True)[0]
    top_nodes = torch.nonzero((p[:, 1] - H).abs() < tol, as_tuple=True)[0]
    bottom_dofs = model.dof_indices(bottom_nodes)
    top_y = model.dof_indices(top_nodes).reshape(-1, 2)[:, 1]

    def prescribed_fn(d):
        pd = torch.cat([bottom_dofs, top_y])
        pv = torch.cat([torch.zeros(bottom_dofs.numel(), dtype=dtype),
                        torch.full((top_y.numel(),), d, dtype=dtype)])
        return pd, pv

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)
    snapshots = []
    d_prev = 0.0
    for d in torch.linspace(0.0, 0.055, 40, dtype=dtype).tolist()[1:]:
        out = adaptive_displacement_solve(
            model, prescribed_fn, kappa, u, d_prev, d, initial_step=(d - d_prev), max_iter=80,
        )
        if out is None or out[3] < d - 1e-12:
            print(f"stopped at u = {out[3] if out else d_prev:.4f} mm")
            break
        result, u, kappa, d_prev = out
        snapshots.append((d, result.damage.max(dim=1).values.detach().clone()))

    d_end, D_end = snapshots[-1]
    D_A, D_B = D_end[on_A], D_end[~on_A]
    print(f"final applied displacement: {d_end:.4f} mm")
    print(f"path A (notched): max D = {D_A.max():.3f}, failed elements = {(D_A > 0.99).sum()}")
    print(f"path B (intact):  max D = {D_B.max():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xA = coh_mid[on_A, 0].numpy()
    xB = coh_mid[~on_A, 0].numpy()
    # Milestone snapshots: damage onset on A, first fully-failed element on A, final state.
    idx_onset = next((i for i, (_, D) in enumerate(snapshots) if D[on_A].max() > 0.05), 0)
    idx_first_fail = next((i for i, (_, D) in enumerate(snapshots) if (D[on_A] > 0.99).any()),
                          len(snapshots) - 1)
    for idx in dict.fromkeys([idx_onset, idx_first_fail, len(snapshots) - 1]):
        d, D = snapshots[idx]
        axes[0].plot(xA, D[on_A].numpy(), "-", label=f"path A, u={d:.3f}")
        axes[1].plot(xB, D[~on_A].numpy(), "--", label=f"path B, u={d:.3f}")
    for ax, title in zip(axes, ("candidate path A (ahead of notch): selected",
                                  "candidate path B (un-notched): unloads")):
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("damage D")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("examples/crack_path_selection.png", dpi=150)
    print("Saved examples/crack_path_selection.png")


if __name__ == "__main__":
    main()
