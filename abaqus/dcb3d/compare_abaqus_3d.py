"""3D DCB validation figure: diffcohesive (8-node cohesive elements, hex bulk) vs. Abaqus
(COH3D8/C3D8 on the identical mesh) vs. beam theory (pre-peak compliance line,
C = 8 a0^3 / (E W h^3), plain E: narrow specimen with free lateral faces).

Run from this directory after run_diffcohesive_3d.py and the Abaqus job + extract_odb_3d.py:
    python compare_abaqus_3d.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

L, ARM, W, A0, E = 60.0, 2.0, 4.0, 20.0, 70000.0


def load_csv(path, dcol="delta", pcol="P"):
    delta, P = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            delta.append(float(row[dcol]))
            P.append(float(row[pcol]))
    return delta, P


def trim_envelope(delta, P):
    """No-op for a monotone displacement-controlled path; retained so a future arc-length
    trace (doubling back through local snap-backs) is trimmed to its envelope like the 2D
    comparison (abaqus/dcb/compare_abaqus.py)."""
    d_max, cut = 0.0, len(delta)
    for i, d in enumerate(delta):
        d_max = max(d_max, d)
        if d < 0.98 * d_max:
            cut = i
            break
    return delta[:cut], P[:cut]


def main():
    delta_v, P_v = load_csv(HERE / "dcb3d_diffcohesive_mu02.csv")
    delta_0, P_0 = load_csv(HERE / "dcb3d_diffcohesive_mu0.csv")
    delta_abq, P_abq = load_csv(HERE / "dcb3d_result.csv")
    delta_fem, P_fem = delta_v, P_v   # headline curve for the peak printout

    compliance = 8.0 * A0 ** 3 / (E * W * ARM ** 3)
    peak_idx = max(range(len(P_fem)), key=lambda i: P_fem[i])

    plt.figure(figsize=(6.5, 5))
    plt.plot(delta_0, P_0, "o-", ms=4, color="C0",
             label="present, unstabilized (exact; to first post-peak pop-in)")
    plt.plot(delta_v, P_v, "-", lw=1.6, color="C0", alpha=0.55,
             label="present, viscous stabilization $\mu=0.2$ (full branch)")
    plt.plot(delta_abq, P_abq, "s--", ms=4, color="C1",
             label="Abaqus *STATIC (COH3D8/C3D8, same mesh)")
    plt.plot([0.0, delta_fem[peak_idx]], [0.0, delta_fem[peak_idx] / compliance], ":",
             color="gray", label="beam theory (pre-peak)")
    plt.xlabel("Opening displacement $\\Delta$")
    plt.ylabel("Reaction load $P$")
    plt.title("3D DCB: present vs. Abaqus COH3D8 (same hex mesh) vs. beam theory")
    plt.legend()
    plt.tight_layout()
    out = HERE / "dcb3d_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    abq_peak = max(P_abq)
    for tag, P in [("unstabilized", P_0), ("mu=0.2", P_v)]:
        pk = max(P)
        print(f"diffcohesive peak ({tag}): {pk:.4f}  vs Abaqus {abq_peak:.4f}  "
              f"({abs(pk - abq_peak) / abq_peak * 100:.2f}%)")
    print(f"end-of-trace P at max opening: present(mu=0.2) {P_v[-1]:.3f} vs Abaqus {P_abq[-1]:.3f}")


if __name__ == "__main__":
    main()
