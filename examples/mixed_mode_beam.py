"""Mesh-level mixed-mode BK validation, replacing literal ENF/MMB contact-fixture
modeling (the current solver has
no contact/no-penetration formulation, which a real 3-point-bend ENF or MMB lever rig needs).

Extends the point-level BK tests (tests/laws/test_bilinear_mixed_mode.py) to a real assembled
mesh: a short, fully-bonded two-arm beam (crack_length=0, so the cohesive ligament starts right
at the loaded end) with a sweep of normal/shear displacement ratios applied directly to the
crack-mouth cohesive node pair. The realized local separation and traction at the first cohesive
element's quadrature point are read back from the converged mesh solve at each step, and the
resulting dissipated energy is compared to the Benzeggagh-Kenane mixed-mode toughness
prediction for the *realized* (not nominal) mode-mix ratio.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.elements.cohesive import _GAUSS2_XI, _local_frame, _relative_displacement_B
from diffcohesive.laws import BilinearMixedModeTSL
from diffcohesive.mesh import build_double_cantilever_mesh
from diffcohesive.solvers import newton_solve

DEFAULT_PARAMS = dict(
    length=2.0,
    arm_height=1.0,
    nx=4,
    ny=2,
    E=1000.0,
    nu=0.3,
    T_max_n=5.0,
    T_max_s=8.0,
    G_c1=1.0,
    G_c2=2.0,
    eta=1.5,
    K=1.0e4,
)


def run_mixed_mode(
    theta_deg,
    length=2.0,
    arm_height=1.0,
    nx=4,
    ny=2,
    E=1000.0,
    nu=0.3,
    T_max_n=5.0,
    T_max_s=8.0,
    G_c1=1.0,
    G_c2=2.0,
    eta=1.5,
    K=1.0e4,
    n_steps=400,
    max_disp=1.5,
):
    """theta_deg=0 -> pure mode I (normal opening); theta_deg=90 -> pure mode II (shear)."""
    mesh = build_double_cantilever_mesh(length, arm_height, crack_length=0.0, nx=nx, ny=ny)
    law = BilinearMixedModeTSL(T_max_n=T_max_n, T_max_s=T_max_s, G_c1=G_c1, G_c2=G_c2, eta=eta, K=K)
    model = CohesiveMeshModel(
        points=mesh.points,
        bulk_elements={"triangle": mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity,
        law=law,
        E=E,
        nu=nu,
    )
    dtype = model.points.dtype

    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    a_bottom, b_bottom, b_top, a_top = model.cohesive_connectivity[0].tolist()
    load_dofs = model.dof_indices(torch.tensor([a_bottom, a_top]))

    theta = math.radians(theta_deg)
    dn_dir, ds_dir = math.cos(theta), math.sin(theta)

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)

    X1, X2 = model.points[a_bottom], model.points[b_bottom]
    R_mat, _ = _local_frame(X1, X2)
    B0 = _relative_displacement_B(_GAUSS2_XI[0])

    dn_hist, ds_hist, Tn_hist, Ts_hist = [], [], [], []
    P_hist, d_hist = [], []
    disps = torch.linspace(0.0, max_disp, n_steps, dtype=dtype)
    for d in disps:
        prescribed_dofs = torch.cat([right_dofs, load_dofs])
        prescribed_values = torch.cat(
            [
                torch.zeros(right_dofs.numel(), dtype=dtype),
                torch.tensor(
                    [-0.5 * d * ds_dir, -0.5 * d * dn_dir, 0.5 * d * ds_dir, 0.5 * d * dn_dir],
                    dtype=dtype,
                ),
            ]
        )
        result = newton_solve(model, prescribed_dofs, prescribed_values, kappa, u0=u, max_iter=100)
        if not result.converged:
            break
        u, kappa = result.u, result.kappa

        dofs0 = model.dof_indices(model.cohesive_connectivity[0])
        delta_u_global = B0 @ u[dofs0]
        delta_local = R_mat @ delta_u_global
        traction, _, _ = law(delta_local, kappa[0, 0].detach())

        dn_hist.append(delta_local[0].item())
        ds_hist.append(delta_local[1].item())
        Tn_hist.append(traction[0].item())
        Ts_hist.append(traction[1].item())
        # Resultant reaction magnitude at the loaded (crack-mouth) pair vs applied separation
        # magnitude -- the mode-II / mixed-mode load-displacement curve.
        R_pair = result.reaction[-4:]
        P_hist.append(0.5 * torch.linalg.norm(R_pair.reshape(2, 2), dim=1).sum().item())
        d_hist.append(d.item())

    dn_t = torch.tensor(dn_hist, dtype=dtype)
    ds_t = torch.tensor(ds_hist, dtype=dtype)
    Tn_t = torch.tensor(Tn_hist, dtype=dtype)
    Ts_t = torch.tensor(Ts_hist, dtype=dtype)
    dissipated = torch.trapezoid(Tn_t, dn_t).item() + torch.trapezoid(Ts_t, ds_t).item()

    # Realized mode-mix at (near-)failure: use the last recorded point.
    dn_f, ds_f = dn_hist[-1], ds_hist[-1]
    mn_f = max(dn_f, 0.0)
    mode_mix = ds_f ** 2 / (ds_f ** 2 + mn_f ** 2 + 1e-12)
    Gc_predicted = G_c1 + (G_c2 - G_c1) * mode_mix ** eta

    return {
        "theta_deg": theta_deg,
        "mode_mix": mode_mix,
        "dissipated": dissipated,
        "Gc_predicted": Gc_predicted,
        "dn": dn_hist,
        "ds": ds_hist,
        "Tn": Tn_hist,
        "Ts": Ts_hist,
        "P": P_hist,
        "d": d_hist,
        "n_converged": len(dn_hist),
    }


def main():
    print(f"{'theta':>6} {'mode_mix':>10} {'dissipated':>12} {'Gc_pred':>10} {'rel.err%':>9}")
    mode_mixes, dissipated_list, predicted_list, curves = [], [], [], {}
    for theta_deg in [0, 15, 30, 45, 60, 75, 90]:
        r = run_mixed_mode(theta_deg, **DEFAULT_PARAMS)
        rel_err = abs(r["dissipated"] - r["Gc_predicted"]) / r["Gc_predicted"] * 100
        print(f"{theta_deg:6d} {r['mode_mix']:10.4f} {r['dissipated']:12.5f} {r['Gc_predicted']:10.5f} {rel_err:9.2f}")
        mode_mixes.append(r["mode_mix"])
        dissipated_list.append(r["dissipated"])
        predicted_list.append(r["Gc_predicted"])
        curves[theta_deg] = r

    mix_line = torch.linspace(0.0, 1.0, 100)
    Gc_line = DEFAULT_PARAMS["G_c1"] + (DEFAULT_PARAMS["G_c2"] - DEFAULT_PARAMS["G_c1"]) * mix_line ** DEFAULT_PARAMS["eta"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) realized effective traction-separation response at the crack-mouth quadrature
    # point, per loading angle: peak + softening + mode-dependent strength. (Specimen-level
    # mode-II LOAD-displacement curves are the pull-out benchmark of the shape study.)
    import math as _m
    for theta_deg, ls in [(0, "-"), (45, "--"), (90, "-.")]:
        r = curves[theta_deg]
        d_eff = [_m.sqrt(max(a, 0.0) ** 2 + b * b) for a, b in zip(r["dn"], r["ds"])]
        # Tension-only normal content: after failure in shear-dominated loading the crack
        # mouth develops contact (negative delta_n, full penalty) -- physically correct, but
        # it is contact, not cohesion, so it is excluded from the cohesive-response plot.
        T_eff = [_m.sqrt(max(a, 0.0) ** 2 + b * b) for a, b in zip(r["Tn"], r["Ts"])]
        label = {0: "pure mode I ($\\theta$=0)", 45: "mixed ($\\theta$=45$^\\circ$)",
                 90: "pure mode II ($\\theta$=90$^\\circ$)"}[theta_deg]
        axes[0].plot(d_eff, T_eff, ls, label=label)
    axes[0].set_xlabel("Effective separation at crack mouth (mm)")
    axes[0].set_ylabel("Effective traction (N/mm$^2$)")
    axes[0].set_title("(a) realized traction-separation, present", fontsize=10)
    axes[0].legend(fontsize=8)
    # (b) dissipated energy vs BK prediction
    axes[1].plot(mix_line, Gc_line, "-", color="gray", label="BK prediction $G_c(B)$ (reference)")
    axes[1].plot(mode_mixes, dissipated_list, "o", ms=7, label="present (mesh-level dissipation)")
    axes[1].set_xlabel("Mode-mix ratio (realized)")
    axes[1].set_ylabel("Dissipated energy (N/mm)")
    axes[1].set_title("(b) toughness vs mode mix", fontsize=10)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("examples/mixed_mode_result.png", dpi=150)
    print("Saved figure to examples/mixed_mode_result.png")


if __name__ == "__main__":
    main()
