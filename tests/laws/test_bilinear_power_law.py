"""Power-law mixed-mode criterion (the standard alternative to BK, e.g. in Camanho-Davila's
own comparisons and Abaqus's *DAMAGE EVOLUTION options): pure-mode limits must recover
G_c1/G_c2 exactly, a proportional mixed-mode path must dissipate the power-law-predicted
toughness, and gradients must stay finite through the fractional powers at the pure-mode
limits (the clamp in the implementation exists for exactly that)."""

import torch

from diffcohesive.laws import BilinearMixedModeTSL


def _power_law(alpha=2.0):
    return BilinearMixedModeTSL(
        T_max_n=50.0, T_max_s=80.0, G_c1=1.0, G_c2=2.0, eta=alpha, K=1.0e5,
        mixed_mode_criterion="power",
    )


def _dissipated_energy_along_path(law, delta_n_fn, delta_s_fn, n_steps, t_max):
    kappa = torch.tensor(0.0, dtype=torch.float64)
    ts = torch.linspace(0.0, t_max, n_steps, dtype=torch.float64)
    Tn, Ts, dn, ds = [], [], [], []
    for t in ts:
        d_n, d_s = delta_n_fn(t), delta_s_fn(t)
        delta_local = torch.stack([d_n, d_s, torch.zeros((), dtype=torch.float64)])
        traction, kappa, _ = law(delta_local, kappa)
        Tn.append(traction[0])
        Ts.append(traction[1])
        dn.append(d_n)
        ds.append(d_s)
    Tn, Ts, dn, ds = map(torch.stack, (Tn, Ts, dn, ds))
    return (torch.trapezoid(Tn, dn) + torch.trapezoid(Ts, ds)).item()


def test_pure_mode_limits_recover_Gc1_and_Gc2():
    law = _power_law(alpha=2.0)
    K = law.K.item()
    zero = lambda t: torch.zeros((), dtype=torch.float64)

    deltaf_1 = 2.0 * law.G_c1.item() / (K * law.T_max_n.item() / K)
    d1 = _dissipated_energy_along_path(law, lambda t: t, zero, 4000, 1.2 * deltaf_1)
    assert abs(d1 - law.G_c1.item()) / law.G_c1.item() < 5e-3

    deltaf_2 = 2.0 * law.G_c2.item() / (K * law.T_max_s.item() / K)
    d2 = _dissipated_energy_along_path(law, zero, lambda t: t, 4000, 1.2 * deltaf_2)
    assert abs(d2 - law.G_c2.item()) / law.G_c2.item() < 5e-3


def test_proportional_mixed_mode_dissipates_power_law_predicted_energy():
    alpha = 2.0
    law = _power_law(alpha=alpha)
    K = law.K.item()
    delta0_n = law.T_max_n.item() / K
    delta0_s = law.T_max_s.item() / K

    # delta_n = delta_s = t stays at mode_mix B = 0.5 throughout.
    B = 0.5
    Gc1, Gc2 = law.G_c1.item(), law.G_c2.item()
    Gc_m = (((1 - B) / Gc1) ** alpha + (B / Gc2) ** alpha) ** (-1.0 / alpha)
    delta0_m = (delta0_n ** 2 + (delta0_s ** 2 - delta0_n ** 2) * B) ** 0.5
    deltaf_m = 2.0 * Gc_m / (K * delta0_m)
    t_max = 1.2 * deltaf_m / 2 ** 0.5

    dissipated = _dissipated_energy_along_path(law, lambda t: t, lambda t: t, 6000, t_max)
    assert abs(dissipated - Gc_m) / Gc_m < 1.5e-2

    # And it must differ measurably from the BK prediction at the same eta, so the test can't
    # silently pass with the wrong criterion wired in.
    Gc_bk = Gc1 + (Gc2 - Gc1) * B ** alpha
    assert abs(Gc_m - Gc_bk) / Gc_bk > 0.05


def test_gradients_finite_through_power_law_at_pure_mode_limit():
    law = _power_law(alpha=2.0)
    kappa = torch.tensor(0.0, dtype=torch.float64)
    # Pure mode I (mode_mix -> 0): the fractional powers must not produce NaN/inf grads.
    delta_local = torch.tensor([2e-3, 0.0, 0.0], dtype=torch.float64)
    traction, kappa, _ = law(delta_local, kappa)
    loss = traction.pow(2).sum()
    loss.backward()
    for name, p in law.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
