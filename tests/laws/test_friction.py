"""Constitutive validation of FrictionalCohesiveTSL against the structural predictions of
Alfano & Sacco, IJNME 68 (2006) 542-582: exact bilinear reduction in pure mode I tension,
Coulomb residual-shear plateau mu*|sigma| after complete decohesion under compression,
dissipation exceeding G_c2 by exactly the frictional work, Coulomb reversal on reverse
sliding, and no damage growth under pure compression. Parameters chosen to satisfy the
paper's proportionality hypothesis so1/sc1 = so2/sc2 exactly."""

import torch

from diffcohesive.laws import FrictionalCohesiveTSL

# so1/sc1 = so2/sc2 by construction (identical mode-I and mode-II sets).
PARAMS = dict(sigma0=3.0, tau0=3.0, G_c1=0.1, G_c2=0.1, K1=1.0e4, K2=1.0e4, mu=0.5)


def _law(**overrides):
    return FrictionalCohesiveTSL(**{**PARAMS, **overrides})


def _state0():
    return torch.zeros(2, dtype=torch.float64)


def test_pure_mode_i_tension_reduces_to_bilinear():
    law = _law()
    sc1 = 2.0 * PARAMS["G_c1"] / PARAMS["sigma0"]
    deltas = torch.linspace(0.0, 1.2 * sc1, 6000, dtype=torch.float64)
    state = _state0()
    tractions = []
    for d in deltas:
        t, state, _ = law(torch.stack([d, torch.zeros((), dtype=torch.float64)]), state)
        tractions.append(t[0])
    tractions = torch.stack(tractions)
    dissipated = torch.trapezoid(tractions, deltas).item()
    assert abs(dissipated - PARAMS["G_c1"]) / PARAMS["G_c1"] < 1e-2
    assert abs(tractions.max().item() - PARAMS["sigma0"]) / PARAMS["sigma0"] < 1e-2
    assert tractions[-1].item() < 1e-2 * PARAMS["sigma0"]


def test_mode_ii_under_compression_reaches_coulomb_plateau_and_dissipates_more_than_Gc2():
    law = _law()
    s1_c = -1.0e-3  # fixed compressive normal separation -> contact stress K1*s1_c = -10
    sc2 = 2.0 * PARAMS["G_c2"] / PARAMS["tau0"]
    s2_end = 3.0 * sc2
    s2s = torch.linspace(0.0, s2_end, 9000, dtype=torch.float64)
    state = _state0()
    taus, sigmas, damages = [], [], []
    for s2 in s2s:
        t, state, D = law(torch.stack([torch.tensor(s1_c, dtype=torch.float64), s2]), state)
        sigmas.append(t[0].item())
        taus.append(t[1])
        damages.append(D.item())
    taus = torch.stack(taus)

    assert damages[-1] > 0.999  # fully decohered
    # Normal response is full-stiffness contact throughout (both REA parts carry K1*s1).
    assert abs(sigmas[-1] - PARAMS["K1"] * s1_c) / abs(PARAMS["K1"] * s1_c) < 1e-6

    # Residual shear = Coulomb plateau mu * |sigma| (paper's central qualitative result).
    plateau_expected = PARAMS["mu"] * abs(PARAMS["K1"] * s1_c)
    assert abs(taus[-1].item() - plateau_expected) / plateau_expected < 2e-2

    # Total dissipation exceeds G_c2 by (approximately) the frictional work on the slip path.
    dissipated = torch.trapezoid(taus, s2s).item() - 0.5 * taus[-1].item() ** 2 / PARAMS["K2"]
    assert dissipated > 1.2 * PARAMS["G_c2"]

    # And with mu = 0 the same path dissipates just G_c2 (Crisfield-only behaviour).
    law0 = _law(mu=0.0)
    state = _state0()
    taus0 = []
    for s2 in s2s:
        t, state, _ = law0(torch.stack([torch.tensor(s1_c, dtype=torch.float64), s2]), state)
        taus0.append(t[1])
    taus0 = torch.stack(taus0)
    dissipated0 = torch.trapezoid(taus0, s2s).item()
    assert abs(dissipated0 - PARAMS["G_c2"]) / PARAMS["G_c2"] < 2e-2
    assert abs(taus0[-1].item()) < 2e-2 * PARAMS["tau0"]


def test_reverse_sliding_flips_coulomb_plateau_sign():
    law = _law()
    s1_c = -1.0e-3
    sc2 = 2.0 * PARAMS["G_c2"] / PARAMS["tau0"]
    state = _state0()
    tau_last = None
    # Slide forward through complete decohesion...
    for s2 in torch.linspace(0.0, 3.0 * sc2, 4000, dtype=torch.float64):
        t, state, _ = law(torch.stack([torch.tensor(s1_c, dtype=torch.float64), s2]), state)
        tau_last = t[1].item()
    plateau = PARAMS["mu"] * abs(PARAMS["K1"] * s1_c)
    assert abs(tau_last - plateau) / plateau < 2e-2
    # ...then slide backwards far enough to reverse the slip direction.
    for s2 in torch.linspace(3.0 * sc2, -3.0 * sc2, 8000, dtype=torch.float64):
        t, state, _ = law(torch.stack([torch.tensor(s1_c, dtype=torch.float64), s2]), state)
    assert abs(t[1].item() + plateau) / plateau < 2e-2  # now -mu*|sigma|


def test_pure_compression_causes_no_damage_and_full_stiffness():
    law = _law()
    state = _state0()
    for s1 in torch.linspace(0.0, -5.0e-3, 200, dtype=torch.float64):
        t, state, D = law(torch.stack([s1, torch.zeros((), dtype=torch.float64)]), state)
        assert D.item() < 1e-9
    assert abs(t[0].item() - PARAMS["K1"] * (-5.0e-3)) / abs(PARAMS["K1"] * 5.0e-3) < 1e-6
