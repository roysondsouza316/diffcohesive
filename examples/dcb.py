"""Real DCB (mode I) mesh validation. Builds a rectangular double-cantilever-beam specimen (two
arms + bonded ligament + pre-crack region), opens it via displacement control (Newton for the
elastic pre-peak, handing off to arc-length once Newton stops converging near the snap-back),
and compares the resulting load-displacement curve to classical DCB beam theory. See
abaqus/dcb/dcb.inp for the independent Abaqus COH2D4 reference model and
abaqus/dcb/compare_abaqus.py for the three-way (diffcohesive / Abaqus / LEFM) comparison figure.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import build_double_cantilever_mesh
from diffcohesive.solvers import arc_length_solve, newton_solve

DEFAULT_PARAMS = dict(
    length=15.0,
    arm_height=1.0,
    crack_length=5.0,
    nx=30,
    ny=4,
    E=1000.0,
    nu=0.3,
    T_max_n=5.0,
    T_max_s=5.0,
    G_c1=0.05,
    G_c2=0.05,
    K=1.0e4,
)


def lefm_compliance(a0: float, E: float, arm_height: float) -> float:
    """delta/P for a symmetric DCB, simple beam theory: delta = 8*P*a0^3/(E*b*h^3),
    with b (out-of-plane width) = 1 for this plane-strain 2D model."""
    return 8.0 * a0 ** 3 / (E * arm_height ** 3)


def build_dcb_model(
    length, arm_height, crack_length, nx, ny, E, nu, T_max_n, T_max_s, G_c1, G_c2, K,
    viscosity=0.0,
):
    mesh = build_double_cantilever_mesh(length, arm_height, crack_length, nx, ny)
    law = BilinearMixedModeTSL(T_max_n=T_max_n, T_max_s=T_max_s, G_c1=G_c1, G_c2=G_c2, K=K,
                               viscosity=viscosity)
    model = CohesiveMeshModel(
        points=mesh.points,
        bulk_elements={"triangle": mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity,
        law=law,
        E=E,
        nu=nu,
    )
    return model, mesh


def run_dcb(
    length=15.0,
    arm_height=1.0,
    crack_length=5.0,
    nx=30,
    ny=4,
    E=1000.0,
    nu=0.3,
    T_max_n=5.0,
    T_max_s=5.0,
    G_c1=0.05,
    G_c2=0.05,
    K=1.0e4,
    n_disp_steps=60,
    max_disp=0.6,
    arc_ds=0.003,
    arc_steps=200,
    viscosity=0.0,
):
    model, mesh = build_dcb_model(
        length, arm_height, crack_length, nx, ny, E, nu, T_max_n, T_max_s, G_c1, G_c2, K,
        viscosity=viscosity,
    )
    if viscosity > 0.0:
        # Viscous damage regularization active: displacement control with adaptive increment
        # cutting traces the whole softening branch directly (no arc-length handoff needed),
        # mirroring the 3D comparison runner (abaqus/dcb3d/run_diffcohesive_3d.py).
        from diffcohesive.solvers import adaptive_displacement_solve

        dtype = model.points.dtype
        right_dofs = model.dof_indices(mesh.right_edge_nodes)
        tip_top_y = model.dof_indices(torch.tensor([mesh.tip_top]))[1]
        tip_bottom_y = model.dof_indices(torch.tensor([mesh.tip_bottom]))[1]

        def prescribed_fn(d):
            pd = torch.cat([right_dofs, tip_top_y.reshape(1), tip_bottom_y.reshape(1)])
            pv = torch.cat([
                torch.zeros(right_dofs.numel(), dtype=dtype),
                torch.tensor([d / 2, -d / 2], dtype=dtype),
            ])
            return pd, pv

        kappa = model.init_history()
        u = torch.zeros(model.n_dof, dtype=dtype)
        delta_list, P_list = [0.0], [0.0]
        d_prev = 0.0
        for d in torch.linspace(0.0, max_disp, n_disp_steps, dtype=dtype)[1:]:
            out = adaptive_displacement_solve(
                model, prescribed_fn, kappa, u, d_prev, float(d),
                initial_step=float(d) - d_prev, max_iter=60,
            )
            if out is None or out[3] < float(d) - 1e-12:
                break
            result, u, kappa, d_prev = out
            P_top, P_bottom = result.reaction[-2].item(), result.reaction[-1].item()
            delta_list.append(float(d))
            P_list.append(0.5 * (P_top - P_bottom))
        return {
            "delta": delta_list,
            "P": P_list,
            "mesh": mesh,
            "model": model,
            "switch_index": None,
        }
    dtype = model.points.dtype

    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    tip_top_dofs = model.dof_indices(torch.tensor([mesh.tip_top]))
    tip_bottom_dofs = model.dof_indices(torch.tensor([mesh.tip_bottom]))
    tip_top_y = tip_top_dofs[1]
    tip_bottom_y = tip_bottom_dofs[1]

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)

    disps = torch.linspace(0.0, max_disp, n_disp_steps, dtype=dtype)
    P_list, delta_list = [], []
    last_converged = None
    switch_index = None
    for i, d in enumerate(disps):
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
            switch_index = i
            break
        u, kappa = result.u, result.kappa
        P_top, P_bottom = result.reaction[-2].item(), result.reaction[-1].item()
        P_list.append(0.5 * (P_top - P_bottom))
        delta_list.append(d.item())
        last_converged = result

    # Hand off to arc-length once Newton stops converging (softening/snap-back regime): release
    # the tip DOFs to be load-controlled (free, driven by lambda*f_hat) while the far end stays
    # clamped, continuing from the last converged displacement-controlled state.
    if switch_index is not None and last_converged is not None:
        f_hat = torch.zeros(model.n_dof, dtype=dtype)
        f_hat[tip_top_y] = 1.0
        f_hat[tip_bottom_y] = -1.0
        lam0 = P_list[-1] if P_list else 0.0
        history = arc_length_solve(
            model,
            fixed_dofs=right_dofs,
            f_hat=f_hat,
            kappa_state=kappa,
            ds=arc_ds,
            n_steps=arc_steps,
            u0=u,
            lam0=lam0,
        )
        for step in history:
            if not step.converged:
                break
            P_list.append(step.lam)
            delta = (step.u[tip_top_y] - step.u[tip_bottom_y]).item()
            delta_list.append(delta)

    return {
        "delta": delta_list,
        "P": P_list,
        "mesh": mesh,
        "model": model,
        "switch_index": switch_index,
    }


def main():
    result = run_dcb(**{})
    delta, P = result["delta"], result["P"]
    a0 = result["mesh"].crack_length
    compliance = lefm_compliance(a0, DEFAULT_PARAMS["E"], DEFAULT_PARAMS["arm_height"])

    # Pre-peak compliance check: use points comfortably before the peak load.
    peak_idx = max(range(len(P)), key=lambda i: P[i])
    check_idx = max(1, peak_idx // 2)
    fem_compliance = delta[check_idx] / P[check_idx]
    rel_err = abs(fem_compliance - compliance) / compliance
    print(f"crack length a0 = {a0}")
    print(f"LEFM compliance delta/P = {compliance:.6e}")
    print(f"FEM pre-peak compliance (index {check_idx}) = {fem_compliance:.6e}")
    print(f"relative error = {rel_err * 100:.2f}%")
    print(f"peak load P_max = {P[peak_idx]:.4f} at delta = {delta[peak_idx]:.5f}")

    plt.figure(figsize=(6, 4.5))
    plt.plot(delta, P, "o-", ms=3, label="diffcohesive (FEM)")
    delta_lin = [0.0, delta[peak_idx]]
    plt.plot(delta_lin, [d / compliance for d in delta_lin], "--", label="LEFM beam theory (pre-peak)")
    plt.xlabel("Opening displacement delta")
    plt.ylabel("Reaction load P")
    plt.title("DCB mode-I load-displacement response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/dcb_result.png", dpi=150)
    print("Saved figure to examples/dcb_result.png")


if __name__ == "__main__":
    main()
