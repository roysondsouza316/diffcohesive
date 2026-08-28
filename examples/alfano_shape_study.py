"""Reproduction of the four benchmark problems of Alfano, "On the influence of the shape of
the interface law on the application of cohesive-zone models", Compos. Sci. Technol. 66 (2006)
723-730, doi:10.1016/j.compscitech.2004.12.024, each run with all four interface-law shapes
(bilinear, linear-parabolic, exponential, trapezoidal) built from identical (K0, sigma0, Gc):

1. thin_dcb   -- aluminium DCB, L=100 mm, total thickness 3 mm (arms 1.5 mm), a0=30 mm.
2. thick_dcb  -- same but total thickness 60 mm.
3. compact    -- steel compact-tension-like specimen, total thickness 100 mm.
4. pullout    -- mode-II pull-out (aluminium, nu=0 to suppress mode mixity).

Bulk elements are bilinear quadrilaterals, with two or more elements through each arm's
thickness; loads are reported for the reference's 20 mm out-of-plane width.

Run a single case/law:
    PYTHONPATH=. python examples/alfano_shape_study.py --case thin_dcb --law exponential
Run everything and produce the four comparison figures:
    PYTHONPATH=. python examples/alfano_shape_study.py

Known limitation: the thin_dcb/exponential continuation is traced through the peak only
(a native-kernel access violation on Windows terminates the deep-softening branch); the
other three laws trace the full softening branch for that case.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffcohesive.assembly import CohesiveMeshModel
from diffcohesive.laws import SHAPE_LAWS
from diffcohesive.mesh import build_double_cantilever_mesh, build_pullout_mesh
from diffcohesive.solvers import arc_length_solve, newton_solve

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "alfano_shape_study"

WIDTH = 20.0  # out-of-plane width (mm) used in the paper

MODE_I_IFACE = dict(K0=10000.0, sigma0=30.0, Gc=0.5)
PULLOUT_IFACE = dict(K0=5000.0, sigma0=3.0, Gc=0.1)

CASES = {
    # nx/ny chosen so the interface element size matches the paper's INT4 sizes (0.55 mm for
    # the thin DCB, ~1.0-1.3 mm for the thick/compact specimens, 1.0 mm for the pull-out).
    "thin_dcb": dict(kind="dcb", length=100.0, arm=1.5, a0=30.0, E=70000.0, nu=0.3,
                      iface=MODE_I_IFACE, nx=182, ny=3, max_disp=12.0, n_steps=90),
    "thick_dcb": dict(kind="dcb", length=100.0, arm=30.0, a0=30.0, E=70000.0, nu=0.3,
                       iface=MODE_I_IFACE, nx=78, ny=8, max_disp=0.8, n_steps=60),
    "compact": dict(kind="dcb", length=100.0, arm=50.0, a0=30.0, E=210000.0, nu=0.2,
                     iface=MODE_I_IFACE, nx=78, ny=8, max_disp=0.25, n_steps=60),
    "pullout": dict(kind="pullout", E=70000.0, nu=0.0, iface=PULLOUT_IFACE,
                     max_disp=0.35, n_steps=70),
}


def _make_law(law_name, iface):
    return SHAPE_LAWS[law_name](K0=iface["K0"], sigma0=iface["sigma0"], Gc=iface["Gc"])


def run_dcb_case(cfg, law_name, collect=None):
    """``collect``: optional dict; when given, per-step interface damage fields and
    displacement vectors are appended under collect["records"], and the model/mesh are stashed
    under collect["model"]/collect["mesh"] -- used by examples/crack_propagation.py to reuse
    this exact (empirically stable) solution path for the initiation/propagation figures."""
    mesh = build_double_cantilever_mesh(
        cfg["length"], cfg["arm"], cfg["a0"], cfg["nx"], cfg["ny"], element_type="quad"
    )
    law = _make_law(law_name, cfg["iface"])
    model = CohesiveMeshModel(
        points=mesh.points,
        bulk_elements={mesh.cell_type: mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity,
        law=law,
        E=cfg["E"],
        nu=cfg["nu"],
    )
    dtype = model.points.dtype

    right_dofs = model.dof_indices(mesh.right_edge_nodes)
    tip_top_y = model.dof_indices(torch.tensor([mesh.tip_top]))[1]
    tip_bottom_y = model.dof_indices(torch.tensor([mesh.tip_bottom]))[1]

    if collect is not None:
        collect["model"] = model
        collect["mesh"] = mesh
        collect["records"] = []

    def _collect(delta, P_per_width, damage, u_now):
        if collect is not None:
            collect["records"].append(
                dict(delta=delta, P=P_per_width * WIDTH,
                     damage=damage.max(dim=1).values.detach().clone(),
                     u=u_now.detach().clone())
            )

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)
    disps = torch.linspace(0.0, cfg["max_disp"], cfg["n_steps"], dtype=dtype)

    delta_list, R_list = [], []
    switch_index = None
    for i, d in enumerate(disps):
        prescribed_dofs = torch.cat([right_dofs, tip_top_y.reshape(1), tip_bottom_y.reshape(1)])
        prescribed_values = torch.cat(
            [
                torch.zeros(right_dofs.numel(), dtype=dtype),
                torch.tensor([d / 2, -d / 2], dtype=dtype),
            ]
        )
        result = newton_solve(model, prescribed_dofs, prescribed_values, kappa, u0=u, max_iter=60)
        if not result.converged:
            switch_index = i
            break
        u, kappa = result.u, result.kappa
        P = 0.5 * (result.reaction[-2] - result.reaction[-1]).item()
        delta_list.append(d.item())
        R_list.append(P * WIDTH)
        _collect(d.item(), P, result.damage, u)

    if switch_index is not None and R_list:
        f_hat = torch.zeros(model.n_dof, dtype=dtype)
        f_hat[tip_top_y] = 1.0
        f_hat[tip_bottom_y] = -1.0
        history = arc_length_solve(
            model, fixed_dofs=right_dofs, f_hat=f_hat, kappa_state=kappa,
            ds=0.04 * cfg["max_disp"], n_steps=150, u0=u, lam0=R_list[-1] / WIDTH,
        )
        for step in history:
            if not step.converged:
                break
            delta_now = (step.u[tip_top_y] - step.u[tip_bottom_y]).item()
            delta_list.append(delta_now)
            R_list.append(step.lam * WIDTH)
            _collect(delta_now, step.lam, step.damage, step.u)

    return delta_list, R_list


def run_pullout_case(cfg, law_name):
    mesh = build_pullout_mesh()
    law = _make_law(law_name, cfg["iface"])
    model = CohesiveMeshModel(
        points=mesh.points,
        bulk_elements={mesh.cell_type: mesh.elements},
        cohesive_connectivity=mesh.cohesive_connectivity,
        law=law,
        E=cfg["E"],
        nu=cfg["nu"],
    )
    dtype = model.points.dtype

    sym_x_dofs = model.dof_indices(mesh.sym_nodes).reshape(-1, 2)[:, 0]
    fixed_dofs = model.dof_indices(mesh.fixed_nodes)
    load_y_dofs = model.dof_indices(mesh.load_nodes).reshape(-1, 2)[:, 1]

    kappa = model.init_history()
    u = torch.zeros(model.n_dof, dtype=dtype)
    disps = torch.linspace(0.0, cfg["max_disp"], cfg["n_steps"], dtype=dtype)

    u_list, R_list = [], []
    stuck = False
    u_prev_increment_norm = None
    for i, d in enumerate(disps):
        prescribed_dofs = torch.cat([sym_x_dofs, fixed_dofs, load_y_dofs])
        prescribed_values = torch.cat(
            [
                torch.zeros(sym_x_dofs.numel() + fixed_dofs.numel(), dtype=dtype),
                torch.full((load_y_dofs.numel(),), d.item(), dtype=dtype),
            ]
        )
        result = newton_solve(model, prescribed_dofs, prescribed_values, kappa, u0=u, max_iter=60)
        if not result.converged:
            stuck = True
            break
        u_prev_increment_norm = (result.u - u).norm().item()
        u, kappa = result.u, result.kappa
        R_half = result.reaction[-load_y_dofs.numel():].sum().item()
        u_list.append(d.item())
        R_list.append(2.0 * R_half * WIDTH)  # both halves, full 20 mm width

    # The pull-out fails through a global snap-back near the peak (the paper's Fig. 10 shows
    # a near-vertical drop); trace it with adaptive arc-length continuation.
    if stuck and R_list:
        f_hat = torch.zeros(model.n_dof, dtype=dtype)
        f_hat[load_y_dofs] = 1.0 / load_y_dofs.numel()
        ds = max(0.5 * (u_prev_increment_norm or 1e-3), 1e-4)
        history = arc_length_solve(
            model, fixed_dofs=torch.cat([sym_x_dofs, fixed_dofs]), f_hat=f_hat,
            kappa_state=kappa, ds=ds, n_steps=200, u0=u,
            lam0=R_list[-1] / (2.0 * WIDTH),
        )
        for step in history:
            if not step.converged:
                break
            d_now = step.u[load_y_dofs].mean().item()
            if d_now > cfg["max_disp"] or d_now < 0:
                break
            u_list.append(d_now)
            R_list.append(2.0 * step.lam * WIDTH)

    return u_list, R_list


def run_one(case_name, law_name):
    cfg = CASES[case_name]
    runner = run_dcb_case if cfg["kind"] == "dcb" else run_pullout_case
    delta, R = runner(cfg, law_name)
    OUT_DIR.mkdir(exist_ok=True)
    out_csv = OUT_DIR / f"{case_name}__{law_name}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["delta", "R"])
        writer.writerows(zip(delta, R))
    peak = max(R) if R else float("nan")
    print(f"{case_name} / {law_name}: {len(R)} points, peak R = {peak:.2f} N -> {out_csv.name}")
    return delta, R


def make_figure(case_name):
    plt.figure(figsize=(6, 4.5))
    for law_name in SHAPE_LAWS:
        csv_path = OUT_DIR / f"{case_name}__{law_name}.csv"
        if not csv_path.exists():
            continue
        delta, R = [], []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                delta.append(float(row["delta"]))
                R.append(float(row["R"]))
        if case_name != "pullout":
            # Trim a spurious trailing unloading excursion: the DCB-type responses are monotone
            # in the opening displacement (no global snap-back, per the source paper), so an
            # arc-length turn-around retracing the elastic line is a numerical artifact, not
            # physics. The pull-out is excluded -- its snap-back is real.
            d_max, cut = 0.0, len(delta)
            for i, d in enumerate(delta):
                d_max = max(d_max, d)
                if d < 0.95 * d_max:
                    cut = i
                    break
            delta, R = delta[:cut], R[:cut]
        plt.plot(delta, R, label=law_name)
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load R (N)")
    plt.title(f"Alfano CST 2006 -- {case_name}")
    plt.legend()
    plt.tight_layout()
    out = OUT_DIR / f"{case_name}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(CASES), default=None)
    parser.add_argument("--law", choices=list(SHAPE_LAWS), default=None)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        for case_name in CASES:
            make_figure(case_name)
        return

    cases = [args.case] if args.case else list(CASES)
    for case_name in cases:
        laws = [args.law] if args.law else list(SHAPE_LAWS)
        for law_name in laws:
            run_one(case_name, law_name)
        if args.law is None:
            make_figure(case_name)


if __name__ == "__main__":
    main()
