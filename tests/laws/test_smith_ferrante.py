"""Verification of SmithFerranteTSL against the intrinsic-CZM formulation of the
COMET-FEniCSx tour (Bleyer): the normalized traction-opening curve must equal the analytical
T/sigma_c = (delta/delta_0)*exp(1 - delta/delta_0), the damage variable must equal
d = 1 - exp(-kappa/delta_0) exactly (their formulation IS our secant formalism for this
envelope), the envelope must integrate to exactly G_c, and the beta shear-coupling must weight
the effective opening as sqrt(<dn>^2 + beta*ds^2)."""

import math

import torch

from diffcohesive.laws import SmithFerranteTSL

GC, SIGMA_C = 0.5, 50.0  # the tour's weak-interface values (N/mm, MPa)


def test_normalized_traction_opening_matches_analytical_curve():
    law = SmithFerranteTSL(Gc=GC, sigma_c=SIGMA_C)
    d0 = GC / (SIGMA_C * math.e)
    kappa = torch.tensor(0.0, dtype=torch.float64)
    xs = torch.linspace(1e-4, 6.0, 400, dtype=torch.float64)  # delta/delta_0
    for x in xs:
        delta = torch.tensor([x.item() * d0, 0.0], dtype=torch.float64)
        T, kappa, _ = law(delta, kappa)
        T_analytic = SIGMA_C * x.item() * math.exp(1.0 - x.item())
        assert abs(T[0].item() - T_analytic) < 1e-6 * SIGMA_C + 1e-3 * abs(T_analytic)


def test_damage_equals_reference_formula_and_energy_equals_Gc():
    law = SmithFerranteTSL(Gc=GC, sigma_c=SIGMA_C)
    d0 = GC / (SIGMA_C * math.e)
    kappa = torch.tensor(0.0, dtype=torch.float64)
    deltas = torch.linspace(0.0, 12.0 * d0, 8000, dtype=torch.float64)
    tractions = []
    for d in deltas:
        T, kappa, D = law(torch.stack([d, torch.zeros((), dtype=torch.float64)]), kappa)
        tractions.append(T[0])
        # d = 1 - exp(-kappa/delta_0), the reference's damage variable, exactly.
        assert abs(D.item() - (1.0 - math.exp(-kappa.item() / d0))) < 1e-6
    dissipated = torch.trapezoid(torch.stack(tractions), deltas).item()
    assert abs(dissipated - GC) / GC < 5e-3  # envelope integrates to G_c (tail truncated at 12 d0)


def test_unloading_follows_reference_secant():
    law = SmithFerranteTSL(Gc=GC, sigma_c=SIGMA_C)
    d0 = GC / (SIGMA_C * math.e)
    K0 = GC / d0 ** 2
    kappa = torch.tensor(0.0, dtype=torch.float64)
    # Load to 2*delta_0 (past the peak), then unload to half of it.
    T, kappa, D = law(torch.tensor([2 * d0, 0.0], dtype=torch.float64), kappa)
    T_unl, kappa, D_unl = law(torch.tensor([d0, 0.0], dtype=torch.float64), kappa)
    # Frozen to within the smoothed history-max transition width (the C1
    # smoothing trades exact freezing for differentiability).
    assert abs(D_unl.item() - D.item()) < 1e-6
    # Reference unloading: T = (1-d) * K0 * delta.
    assert abs(T_unl[0].item() - (1.0 - D.item()) * K0 * d0) < 1e-6 * SIGMA_C


def test_beta_coupling_weights_shear():
    beta = 2.0  # the tour's coupling coefficient
    law = SmithFerranteTSL(Gc=GC, sigma_c=SIGMA_C, beta=beta)
    d0 = GC / (SIGMA_C * math.e)
    # Pure shear s: effective delta = sqrt(beta)*s -> damage onset earlier than for beta=1.
    kappa1 = torch.tensor(0.0, dtype=torch.float64)
    kappa2 = torch.tensor(0.0, dtype=torch.float64)
    s = 2.0 * d0
    law1 = SmithFerranteTSL(Gc=GC, sigma_c=SIGMA_C, beta=1.0)
    _, _, D1 = law1(torch.tensor([0.0, s], dtype=torch.float64), kappa1)
    _, _, D2 = law(torch.tensor([0.0, s], dtype=torch.float64), kappa2)
    assert D2.item() > D1.item()  # stronger coupling -> more damage at equal sliding