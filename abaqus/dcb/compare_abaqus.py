"""Three-way DCB validation figure: diffcohesive (autograd FEM) vs. Abaqus (COH2D4 reference,
same mesh) vs. classical LEFM beam theory.

Run from repo root: PYTHONPATH=. python abaqus/dcb/compare_abaqus.py
(requires abaqus/dcb/dcb_abaqus_result.csv already produced by extract_odb.py under Abaqus's
own embedded Python -- see that file's docstring).
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.dcb import DEFAULT_PARAMS, lefm_compliance, run_dcb

HERE = Path(__file__).resolve().parent


def load_abaqus_csv(path):
    delta, P = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            delta.append(float(row["delta"]))
            P.append(float(row["P"]))
    return delta, P


def main():
    fem = run_dcb(**{})
    delta_fem, P_fem = fem["delta"], fem["P"]
    # Stabilized run: small viscous damage regularization traces the full softening branch
    # under displacement control (same device as the 3D comparison); mu kept small so the
    # added artificial toughness stays minor.
    fem_v = run_dcb(viscosity=0.1, max_disp=0.9, n_disp_steps=90)
    delta_v, P_v = fem_v["delta"], fem_v["P"]
    # Show the softening ENVELOPE: our arc-length traces the serrated equilibrium path through
    # each local element snap-back (the path doubles back in delta); both Abaqus solution
    # methods (automatic *STATIC incrementation and RIKS) step over those, so for a like-for-
    # like envelope comparison the doubling-back segments are trimmed here (the full serrated
    # path is retained in examples/dcb_result.png).
    d_max, cut = 0.0, len(delta_fem)
    for i, d in enumerate(delta_fem):
        d_max = max(d_max, d)
        if d < 0.98 * d_max:
            cut = i
            break
    delta_fem, P_fem = delta_fem[:cut], P_fem[:cut]

    # Cache both computed curves so figure styling can be redone without re-solving
    # (used by paper/figures_notebook.ipynb).
    for name, dd, pp in [("dcb_diffcohesive_envelope.csv", delta_fem, P_fem),
                          ("dcb_diffcohesive_mu01.csv", delta_v, P_v)]:
        with open(HERE / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["delta", "P"])
            w.writerows(zip(dd, pp))

    abq_csv = HERE / "dcb_abaqus_result.csv"
    delta_abq, P_abq = load_abaqus_csv(abq_csv)

    a0 = fem["mesh"].crack_length
    compliance = lefm_compliance(a0, DEFAULT_PARAMS["E"], DEFAULT_PARAMS["arm_height"])
    peak_idx = max(range(len(P_fem)), key=lambda i: P_fem[i])
    delta_lin = [0.0, delta_fem[peak_idx]]

    plt.figure(figsize=(6.5, 5))
    plt.plot(delta_fem, P_fem, "o-", ms=3, color="C0",
             label="present, arc-length envelope (exact)")
    plt.plot(delta_v, P_v, "-", lw=1.6, color="C0", alpha=0.55,
             label="present, viscous stabilization $\mu=0.1$ (full branch)")
    plt.plot(delta_abq, P_abq, "s--", ms=4, color="C1",
             label="Abaqus *STATIC (COH2D4, same mesh)")
    riks_csv = HERE / "dcb_riks_job_result.csv"
    if riks_csv.exists():
        delta_rk, P_rk = load_abaqus_csv(riks_csv)
        plt.plot(delta_rk, P_rk, "^:", ms=4, color="C2",
                  label="Abaqus *STATIC, RIKS (same mesh)")
    plt.plot(delta_lin, [d / compliance for d in delta_lin], ":", color="gray",
              label="LEFM beam theory (pre-peak)")
    plt.xlabel("Opening displacement delta")
    plt.ylabel("Reaction load P")
    plt.title("DCB mode-I validation: present vs. Abaqus (2 methods) vs. LEFM")
    plt.legend()
    plt.tight_layout()
    out = HERE / "dcb_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    fem_peak = P_fem[peak_idx]
    abq_peak = max(P_abq)
    rel_err = abs(fem_peak - abq_peak) / abq_peak
    print(f"diffcohesive peak load: {fem_peak:.4f} at delta={delta_fem[peak_idx]:.4f}")
    print(f"Abaqus peak load:       {abq_peak:.4f}")
    print(f"relative peak-load difference: {rel_err * 100:.2f}%")


if __name__ == "__main__":
    main()
