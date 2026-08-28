"""Single cohesive-point validation: recover T(delta), verify irreversibility,
and check dissipated energy == G_c to tolerance."""

import torch

from diffcohesive.laws import BilinearMixedModeTSL


def _pure_mode_i_law():
    return BilinearMixedModeTSL(
        T_max_n=50.0, T_max_s=50.0, G_c1=1.0, G_c2=1.0, eta=1.0, K=1.0e5,
    )


def test_monotonic_loading_dissipates_Gc():
    law = _pure_mode_i_law()
    delta0 = law.T_max_n.item() / law.K.item()
    deltaf = 2.0 * law.G_c1.item() / (law.K.item() * delta0)

    n_steps = 4000
    deltas = torch.linspace(0.0, 1.2 * deltaf, n_steps, dtype=torch.float64)
    kappa = torch.tensor(0.0, dtype=torch.float64)

    tractions = []
    for d in deltas:
        delta_local = torch.stack([d, torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)])
        traction, kappa, damage = law(delta_local, kappa)
        tractions.append(traction[0].item())

    tractions = torch.tensor(tractions, dtype=torch.float64)
    dissipated = torch.trapezoid(tractions, deltas).item()

    assert abs(dissipated - law.G_c1.item()) / law.G_c1.item() < 5e-3
    # Fully failed: traction back to (near) zero past deltaf.
    assert abs(tractions[-1].item()) < 1e-2 * law.T_max_n.item()


def test_unload_reload_follows_secant_with_frozen_damage():
    law = _pure_mode_i_law()
    delta0 = law.T_max_n.item() / law.K.item()
    deltaf = 2.0 * law.G_c1.item() / (law.K.item() * delta0)

    kappa = torch.tensor(0.0, dtype=torch.float64)

    def step(d_val):
        nonlocal kappa
        delta_local = torch.tensor([d_val, 0.0, 0.0], dtype=torch.float64)
        traction, kappa, damage = law(delta_local, kappa)
        return traction[0].item(), damage.item()

    # Load partway into softening.
    d_partial = 0.5 * (delta0 + deltaf)
    T_partial, D_partial = step(d_partial)
    assert D_partial > 0.0 and D_partial < 1.0

    # Unload to a smaller separation: traction must follow the *secant* (1-D)*K*delta with
    # D frozen at D_partial, i.e. be purely elastic (no further damage growth). Tolerance is
    # set by the smoothed-max/-Macaulay transition width, not exact equality.
    d_unload = 0.3 * d_partial
    T_unload, D_unload = step(d_unload)
    assert abs(D_unload - D_partial) < 1e-4
    expected_T_unload = (1.0 - D_partial) * law.K.item() * d_unload
    assert abs(T_unload - expected_T_unload) / expected_T_unload < 1e-4

    # Reload back to d_partial along the same secant, damage still frozen.
    T_reload, D_reload = step(d_partial)
    assert abs(D_reload - D_partial) < 1e-4
    assert abs(T_reload - T_partial) / T_partial < 1e-4

    # Push past the previous max: damage must resume growing (monotone history).
    d_beyond = d_partial + 0.1 * (deltaf - delta0)
    _, D_beyond = step(d_beyond)
    assert D_beyond > D_partial


def test_damage_monotone_and_bounded():
    law = _pure_mode_i_law()
    delta0 = law.T_max_n.item() / law.K.item()
    deltaf = 2.0 * law.G_c1.item() / (law.K.item() * delta0)

    kappa = torch.tensor(0.0, dtype=torch.float64)
    prev_damage = 0.0
    for d in torch.linspace(0.0, 1.5 * deltaf, 200, dtype=torch.float64):
        delta_local = torch.tensor([d.item(), 0.0, 0.0], dtype=torch.float64)
        _, kappa, damage = law(delta_local, kappa)
        assert damage.item() >= prev_damage - 1e-9
        assert -1e-9 <= damage.item() <= 1.0 + 1e-9
        prev_damage = damage.item()


def test_gradients_finite_through_full_loading_history():
    law = BilinearMixedModeTSL(T_max_n=50.0, T_max_s=50.0, G_c1=1.0, G_c2=1.0, eta=1.0, K=1.0e5)
    kappa = torch.tensor(0.0, dtype=torch.float64)
    loss = torch.tensor(0.0, dtype=torch.float64)
    for d in torch.linspace(0.0, 1.0, 50, dtype=torch.float64):
        delta_local = torch.stack([d, torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)])
        traction, kappa, damage = law(delta_local, kappa.detach())
        loss = loss + traction[0] ** 2
    loss.backward()
    for p in law.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
