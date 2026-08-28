"""Crack initiation and propagation, made visible and checked quantitatively (addressing the
"crack initiation, propagation should be shown clearly" gap): the aluminium DCB of Alfano CST
2006 is opened under displacement control while the damage field along the bonded ligament is
recorded at every step. Produces:

1. crack_propagation_curve.png    -- load-displacement curve with the numerically detected
                                     initiation point (first fully failed cohesive element)
                                     and propagation stages marked;
2. crack_propagation_profiles.png -- damage profiles D(x) along the ligament at selected
                                     steps: the process zone forms at the pre-crack tip
                                     (initiation), then translates along the ligament
                                     (propagation);
3. crack_propagation_length.png   -- crack length a vs. applied opening, compared with the
                                     LEFM propagation prediction a(delta) = sqrt(delta/8) *
                                     (12*E'*h^3/Gc)^(1/4) (corrected-modulus beam theory,
                                     E' = E/(1-nu^2) plane strain);
4. crack_propagation_mesh.png     -- deformed meshes (true scale) at initiation, mid
                                     propagation, and near complete delamination, interface
                                     colored by damage.

Run: PYTHONPATH=. python examples/crack_propagation.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from examples.alfano_shape_study import CASES, run_dcb_case

# Alfano CST 2006 thin-DCB properties (mm, N) -- must stay consistent with
# alfano_shape_study.CASES["thin_dcb"] / MODE_I_IFACE, whose exact solution path (Newton with
# increment handling + adaptive Crisfield arc-length through the element pop-ins) is reused
# here verbatim: torch CPU arithmetic is deterministic, and that path is the one empirically
# unaffected by the known native worker-thread crash (see alfano_shape_study.py's docstring).
ARM, A0 = 1.5, 30.0
E, NU = 70000.0, 0.3
GC = 0.5
FAILED = 0.99  # damage threshold defining the crack front


def run():
    collect = {}
    run_dcb_case(CASES["thin_dcb"], "bilinear", collect=collect)
    model, mesh = collect["model"], collect["mesh"]

    # x-position of each bonded cohesive element's midpoint (for damage profiles / crack front).
    coh_x = model.points[model.cohesive_connectivity[:, :2]].mean(dim=1)[:, 0]

    records = []
    delta_max = 0.0
    for rec in collect["records"]:
        # Drop any trailing arc-length turn-around (the DCB response is monotone in opening).
        delta_max = max(delta_max, rec["delta"])
        if rec["delta"] < 0.95 * delta_max:
            break
        failed = rec["damage"] >= FAILED
        crack_len = A0 if not failed.any() else max(A0, coh_x[failed].max().item())
        records.append(
            dict(delta=rec["delta"], P=rec["P"], damage=rec["damage"], crack=crack_len, u=rec["u"])
        )
    return records, coh_x, mesh, model


def lefm_crack_length(delta):
    E_prime = E / (1.0 - NU ** 2)
    return (delta / 8.0) ** 0.5 * (12.0 * E_prime * ARM ** 3 / GC) ** 0.25


def main():
    records, coh_x, mesh, model = run()
    deltas = [r["delta"] for r in records]
    loads = [r["P"] for r in records]
    cracks = [r["crack"] for r in records]

    init_idx = next((i for i, r in enumerate(records) if r["crack"] > A0), None)
    stage_idxs = [init_idx, (init_idx + len(records) - 1) // 2, len(records) - 1]

    # -- 1. load-displacement with initiation/propagation markers --
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(deltas, loads, "-", lw=1.5)
    labels = ["initiation", "propagation", "late propagation"]
    for idx, lab in zip(stage_idxs, labels):
        plt.plot(deltas[idx], loads[idx], "o", ms=8, label=f"{lab}: a = {cracks[idx]:.1f} mm")
    plt.xlabel("Applied opening $\\Delta$ (mm)")
    plt.ylabel("Load R (N)")
    plt.title("DCB: crack initiation and propagation on the global response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/crack_propagation_curve.png", dpi=150)

    # -- 2. damage profiles along the ligament --
    plt.figure(figsize=(6.5, 4.5))
    n_profiles = 8
    prof_idxs = [round(i * (len(records) - 1) / (n_profiles - 1)) for i in range(n_profiles)]
    for idx in prof_idxs:
        plt.plot(coh_x.numpy(), records[idx]["damage"].numpy(), "-",
                  label=f"$\\Delta$={deltas[idx]:.1f} mm")
    plt.axvline(A0, color="gray", ls=":", lw=1)
    plt.text(A0 + 0.5, 0.05, "initial crack tip", fontsize=8, color="gray")
    plt.xlabel("Position along interface x (mm)")
    plt.ylabel("Damage D")
    plt.title("Process zone forming (initiation) and translating (propagation)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig("examples/crack_propagation_profiles.png", dpi=150)

    # -- 3. crack length vs opening, against LEFM --
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(deltas, cracks, "o-", ms=3, label="diffcohesive (damage front, D >= 0.99)")
    prop = [d for d in deltas if lefm_crack_length(d) >= A0]
    plt.plot(prop, [lefm_crack_length(d) for d in prop], "--", label="LEFM beam theory")
    plt.xlabel("Applied opening $\\Delta$ (mm)")
    plt.ylabel("Crack length a (mm)")
    plt.title("Crack propagation: cohesive front vs. LEFM")
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/crack_propagation_length.png", dpi=150)

    # -- 4. deformed meshes colored by damage --
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    elements = mesh.elements
    for ax, idx, lab in zip(axes, stage_idxs, labels):
        rec = records[idx]
        pts = (model.points + rec["u"].reshape(-1, 2)).numpy()
        segs = []
        for elem in elements:
            ids = elem.tolist()
            for a, b in zip(ids, ids[1:] + ids[:1]):
                segs.append([pts[a], pts[b]])
        ax.add_collection(LineCollection(segs, colors="lightsteelblue", lw=0.4))
        sc = ax.scatter(
            coh_x.numpy(),
            [pts[model.cohesive_connectivity[i, 0], 1] for i in range(model.n_coh)],
            c=rec["damage"].numpy(), cmap="inferno_r", vmin=0, vmax=1, s=14, zorder=3,
        )
        ax.set_title(f"{lab}:  $\\Delta$={rec['delta']:.1f} mm,  a={rec['crack']:.1f} mm", fontsize=9)
        ax.set_aspect("equal")
        ax.autoscale()
    fig.colorbar(sc, ax=axes, label="interface damage D", shrink=0.7)
    plt.savefig("examples/crack_propagation_mesh.png", dpi=150)

    print(f"initiation at Delta = {deltas[init_idx]:.2f} mm (peak load {max(loads):.1f} N)")
    print(f"final crack length = {cracks[-1]:.1f} mm at Delta = {deltas[-1]:.1f} mm "
          f"(LEFM: {lefm_crack_length(deltas[-1]):.1f} mm)")
    print("Saved 4 figures to examples/crack_propagation_*.png")


if __name__ == "__main__":
    main()
