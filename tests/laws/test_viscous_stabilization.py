"""Viscous damage regularization (the *DAMAGE STABILIZATION analogue) on the bilinear law:
(1) viscosity=0 is bit-identical to the rate-independent law; (2) with viscosity > 0 the
relaxed damage lags the instantaneous target on an abrupt jump and converges to it under
sustained loading; (3) the regularized law still dissipates ~G_c over a slow full opening."""

import torch

from diffcohesive.laws import BilinearMixedModeTSL

P = dict(T_max_n=5.0, T_max_s=5.0, G_c1=0.05, G_c2=0.05, eta=1.0, K=1.0e4)


def _open(law, deltas):
    state_dim = getattr(law, "state_dim", 1)
    kappa = torch.zeros(() if state_dim == 1 else (state_dim,), dtype=torch.float64)
    out = []
    for d in deltas:
        T, kappa, D = law(torch.tensor([d, 0.0], dtype=torch.float64), kappa)
        out.append((T[0].item(), D.item()))
    return out


def test_zero_viscosity_identical():
    base = BilinearMixedModeTSL(**P)
    visc0 = BilinearMixedModeTSL(**P, viscosity=0.0)
    assert getattr(visc0, "state_dim", 1) == 1
    deltas = torch.linspace(1e-4, 0.015, 30).tolist()
    for (t1, d1), (t2, d2) in zip(_open(base, deltas), _open(visc0, deltas)):
        assert t1 == t2 and d1 == d2


def test_viscous_damage_lags_and_converges():
    mu = 0.5
    law = BilinearMixedModeTSL(**P, viscosity=mu)
    assert law.state_dim == 2
    d0 = P["T_max_n"] / P["K"]
    # abrupt jump deep into softening, then hold: D_v must lag, then relax toward the target
    hold = [8.0 * d0] * 12
    res = _open(law, hold)
    base = BilinearMixedModeTSL(**P)
    D_target = _open(base, [8.0 * d0])[0][1]
    D_first, D_last = res[0][1], res[-1][1]
    assert D_first < D_target  # lags on the jump
    assert abs(D_last - D_target) < 1e-3  # converges under sustained load
    assert all(res[i][1] <= res[i + 1][1] + 1e-12 for i in range(len(res) - 1))  # monotone


def test_viscous_dissipation_close_to_gc_when_slow():
    # many small steps => viscous lag negligible => dissipated energy ~ G_c
    law = BilinearMixedModeTSL(**P, viscosity=0.05)
    d0, Gc = P["T_max_n"] / P["K"], P["G_c1"]
    deltaf = 2.0 * Gc / P["T_max_n"]
    deltas = torch.linspace(1e-6, 1.2 * deltaf, 400).tolist()
    out = _open(law, deltas)
    W = 0.0
    prev_d, prev_t = 0.0, 0.0
    for d, (t, _) in zip(deltas, out):
        W += 0.5 * (t + prev_t) * (d - prev_d)
        prev_d, prev_t = d, t
    assert abs(W - Gc) / Gc < 0.05
