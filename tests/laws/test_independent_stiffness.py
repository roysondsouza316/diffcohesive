"""Independent normal/shear penalty stiffness on the bilinear mixed-mode law:
(1) K_s=None and K_s=K give the same response (the energy-based mode mixity reduces to the
displacement-based one); (2) pure mode II sees stiffness K_s, peak T_max_s, and dissipates
exactly G_c2; (3) pure mode I is unaffected by K_s; (4) mixed-mode dissipation under
proportional loading follows the BK criterion at the energy-based mode ratio."""

import torch

from diffcohesive.laws import BilinearMixedModeTSL

P = dict(T_max_n=5.0, T_max_s=7.0, G_c1=0.05, G_c2=0.09, eta=1.5, K=1.0e4)


def _sweep(law, path):
    """Apply a separation path [(dn, ds), ...]; return per-step (Tn, Ts, D)."""
    kappa = torch.zeros((), dtype=torch.float64)
    out = []
    for dn, ds in path:
        T, kappa, D = law(torch.tensor([dn, ds], dtype=torch.float64), kappa)
        out.append((T[0].item(), T[1].item(), D.item()))
    return out


def _dissipated(path, results):
    W, prev = 0.0, (0.0, 0.0, 0.0, 0.0)
    for (dn, ds), (tn, ts, _) in zip(path, results):
        pdn, pds, ptn, pts = prev
        W += 0.5 * (tn + ptn) * (dn - pdn) + 0.5 * (ts + pts) * (ds - pds)
        prev = (dn, ds, tn, ts)
    return W


def test_ks_equal_matches_single_stiffness():
    base = BilinearMixedModeTSL(**P)
    same = BilinearMixedModeTSL(**P, K_s=P["K"])
    t = torch.linspace(1e-5, 3e-3, 40).tolist()
    path = [(x, 0.7 * x) for x in t]
    for (a1, b1, c1), (a2, b2, c2) in zip(_sweep(base, path), _sweep(same, path)):
        assert abs(a1 - a2) < 1e-10 and abs(b1 - b2) < 1e-10 and abs(c1 - c2) < 1e-12


def test_pure_mode2_uses_ks_and_dissipates_gc2():
    K_s = P["K"] / 3.0
    law = BilinearMixedModeTSL(**P, K_s=K_s)
    d0s = P["T_max_s"] / K_s
    dff = 2.0 * P["G_c2"] / P["T_max_s"]
    # elastic range: shear stiffness is K_s
    T, _, D = law(torch.tensor([0.0, 0.4 * d0s], dtype=torch.float64),
                  torch.zeros((), dtype=torch.float64))
    assert abs(T[1].item() - K_s * 0.4 * d0s) / (K_s * d0s) < 1e-6
    assert D.item() < 1e-9
    # full opening: peak ~ T_max_s, dissipation ~ G_c2
    ds_vals = torch.linspace(1e-6, 1.3 * dff, 500).tolist()
    path = [(0.0, x) for x in ds_vals]
    res = _sweep(law, path)
    peak = max(r[1] for r in res)
    assert abs(peak - P["T_max_s"]) / P["T_max_s"] < 0.02
    W = _dissipated(path, res)
    assert abs(W - P["G_c2"]) / P["G_c2"] < 0.03


def test_pure_mode1_independent_of_ks():
    base = BilinearMixedModeTSL(**P)
    ks = BilinearMixedModeTSL(**P, K_s=P["K"] / 5.0)
    dff = 2.0 * P["G_c1"] / P["T_max_n"]
    path = [(x, 0.0) for x in torch.linspace(1e-6, 1.2 * dff, 200).tolist()]
    for (a1, _, c1), (a2, _, c2) in zip(_sweep(base, path), _sweep(ks, path)):
        # tolerance reflects the eps_shear regularization leak into the mode-mix ratio
        # (a ~1e-4 relative effect on tractions of order T_max_n), not a formulation
        # difference
        assert abs(a1 - a2) < 1e-3 and abs(c1 - c2) < 1e-4


def test_mixed_mode_dissipation_follows_bk_with_unequal_stiffness():
    K, K_s = P["K"], P["K"] / 3.0
    law = BilinearMixedModeTSL(**P, K_s=K_s)
    # proportional separation path dn = ds: energy mode ratio is constant along the path
    B = K_s / (K + K_s)  # with dn = ds: B = K_s ds^2 / (K dn^2 + K_s ds^2)
    G_c_expected = P["G_c1"] + (P["G_c2"] - P["G_c1"]) * B ** P["eta"]
    t = torch.linspace(1e-6, 0.02, 800).tolist()
    path = [(x, x) for x in t]
    res = _sweep(law, path)
    assert res[-1][2] > 0.999  # fully failed at the end of the sweep
    W = _dissipated(path, res)
    assert abs(W - G_c_expected) / G_c_expected < 0.04
