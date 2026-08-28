"""Mesh/process-zone convergence study: DCB peak load must converge as element
size drops below the cohesive zone length l_cz ~= E*G_c1/T_max_n^2, and be visibly unconverged
above it. Uses only the Newton (no arc-length) phase of the DCB analysis -- confirmed in
examples/dcb.py to already capture the peak load cleanly before Newton stops converging, so a
full arc-length trace isn't needed per mesh density here.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from examples.dcb import DEFAULT_PARAMS, build_dcb_model
from diffcohesive.solvers import newton_solve


def peak_load_for_mesh(nx, ny, n_disp_steps=60, max_disp=0.6):
    params = {k: v for k, v in DEFAULT_PARAMS.items() if k not in ("nx", "ny")}
    model, mesh = build_dcb_model(nx=nx, ny=ny, **params)
    dtype = model.points.dtype

    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    tip_top_dofs = model.dof_indices(torch.tensor([mesh.tip_top]))
    tip_bottom_dofs = model.dof_indices(torch.tensor([mesh.tip_bottom]))
    tip_top_y, tip_bottom_y = tip_top_dofs[1], tip_bottom_dofs[1]

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)
    disps = torch.linspace(0.0, max_disp, n_disp_steps, dtype=dtype)
    P_list = []
    for d in disps:
        prescribed_dofs = torch.cat([right_dofs, tip_top_y.reshape(1), tip_bottom_y.reshape(1)])
        prescribed_values = torch.cat(
            [
                torch.zeros(right_dofs.numel(), dtype=dtype),
                torch.tensor([d / 2], dtype=dtype),
                torch.tensor([-d / 2], dtype=dtype),
            ]
        )
        result = newton_solve(model, prescribed_dofs, prescribed_values, kappa, u0=u)
        if not result.converged:
            break
        u, kappa = result.u, result.kappa
        P_top, P_bottom = result.reaction[-2].item(), result.reaction[-1].item()
        P_list.append(0.5 * (P_top - P_bottom))
    return (max(P_list) if P_list else float("nan")), mesh.crack_length


def main():
    E, G_c1, T_max_n = DEFAULT_PARAMS["E"], DEFAULT_PARAMS["G_c1"], DEFAULT_PARAMS["T_max_n"]
    l_cz = E * G_c1 / T_max_n ** 2
    length, ny = DEFAULT_PARAMS["length"], DEFAULT_PARAMS["ny"]
    print(f"cohesive zone length l_cz = {l_cz:.4f}")

    nx_values = [5, 8, 12, 18, 24, 30, 45, 60]
    ratios, peaks = [], []
    for nx in nx_values:
        dx = length / nx
        peak, a0 = peak_load_for_mesh(nx, ny)
        ratios.append(dx / l_cz)
        peaks.append(peak)
        print(f"nx={nx:3d} dx={dx:.4f} dx/l_cz={dx / l_cz:.3f} a0(actual)={a0:.3f} peak={peak:.4f}")

    plt.figure(figsize=(6, 4.5))
    plt.plot(ratios, peaks, "o-")
    plt.axvline(1.0, color="gray", ls="--", label="element size = cohesive zone length")
    plt.xlabel("element size / cohesive zone length (dx / l_cz)")
    plt.ylabel("Peak load P_max")
    plt.title("DCB mesh / process-zone convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/convergence_result.png", dpi=150)
    print("Saved figure to examples/convergence_result.png")


if __name__ == "__main__":
    main()
