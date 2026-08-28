"""Constitutive demonstration of the Alfano-Sacco damage+friction law: mode-II response under
fixed normal compression for several friction coefficients. Shows the paper's central feature:
the response transitions from the cohesive bilinear envelope to the Coulomb residual plateau
mu*|sigma_contact| as decohesion completes -- with mu = 0 recovering the frictionless
Crisfield law (zero residual). Saves examples/friction_demo.png."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffcohesive.laws import FrictionalCohesiveTSL

PARAMS = dict(sigma0=3.0, tau0=3.0, G_c1=0.1, G_c2=0.1, K1=1.0e4, K2=1.0e4)
S1_COMPRESSION = -1.0e-3  # contact pressure K1*|s1| = 10 N/mm^2


def response(mu, s2_max, n=4000):
    law = FrictionalCohesiveTSL(mu=mu, **PARAMS)
    state = torch.zeros(2, dtype=torch.float64)
    s2s = torch.linspace(0.0, s2_max, n, dtype=torch.float64)
    taus = []
    for s2 in s2s:
        t, state, _ = law(torch.stack([torch.tensor(S1_COMPRESSION, dtype=torch.float64), s2]), state)
        taus.append(t[1].item())
    return s2s.numpy(), taus


def main():
    sc2 = 2.0 * PARAMS["G_c2"] / PARAMS["tau0"]
    plt.figure(figsize=(6.5, 4.5))
    for mu in (0.0, 0.25, 0.5):
        s2, tau = response(mu, 3.0 * sc2)
        plt.plot(s2, tau, label=f"$\\mu$ = {mu}")
        plateau = mu * abs(PARAMS["K1"] * S1_COMPRESSION)
        if mu > 0:
            plt.axhline(plateau, color="gray", lw=0.6, ls=":")
    plt.text(2.6 * sc2, 0.15, "$\\mu\\,|\\sigma|$ plateaus", fontsize=8, color="gray")
    plt.xlabel("Tangential separation $s_2$ (mm)")
    plt.ylabel("Shear traction $\\tau$ (N/mm$^2$)")
    plt.title("Damage + friction under normal compression (Alfano-Sacco model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/friction_demo.png", dpi=150)
    print("Saved examples/friction_demo.png")


if __name__ == "__main__":
    main()
