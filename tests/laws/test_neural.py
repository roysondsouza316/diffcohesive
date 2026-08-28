"""Neural-law validation: the monotone neural law's built-in
admissibility (D in [0,1], dD/dkappa >= 0, non-negative dissipation via frozen-D secant
unloading), and recovery of a non-standard (non-bilinear) damage-vs-history shape."""

import torch

from diffcohesive.laws import NeuralMonotoneTSL


def test_damage_bounded_and_monotone_by_construction():
    torch.manual_seed(1)
    law = NeuralMonotoneTSL(K=1.0e4, hidden=8)
    kappas = torch.linspace(0.0, 5.0, 500, dtype=torch.float64)
    D = law.damage_net(kappas)

    assert torch.all(D >= -1e-12)
    assert torch.all(D <= 1.0 + 1e-12)
    assert torch.allclose(D[0], torch.zeros((), dtype=torch.float64), atol=1e-10)
    assert torch.all(D[1:] - D[:-1] >= -1e-9)  # monotone non-decreasing


def test_unloading_follows_frozen_damage_secant():
    torch.manual_seed(2)
    law = NeuralMonotoneTSL(K=1.0e4, hidden=8)

    kappa = torch.tensor(0.0, dtype=torch.float64)
    delta_partial = torch.tensor([0.03, 0.0, 0.0], dtype=torch.float64)
    _, kappa, D_partial = law(delta_partial, kappa)

    delta_unload = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
    traction_unload, kappa2, D_unload = law(delta_unload, kappa)

    assert torch.allclose(D_unload, D_partial, atol=1e-9)
    expected = (1.0 - D_partial) * law.K * delta_unload[0]
    assert torch.allclose(traction_unload[0], expected, atol=1e-8)


def test_recovers_nonstandard_damage_shape():
    torch.manual_seed(3)

    kappa_f = 1.0

    def D_target(kappa):
        # Smooth cubic-in-exponent growth to saturation -- not the bilinear law's rational
        # D(kappa) shape, but with no hard corner (a hard corner is unfittable *exactly* by a
        # smooth tanh network by construction, which is a property of smoothness, not a bug).
        x = kappa.clamp_min(0.0) / kappa_f
        return 1.0 - torch.exp(-3.0 * x.pow(3))

    kappas_train = torch.linspace(0.0, 1.3, 200, dtype=torch.float64)
    targets = D_target(kappas_train)

    law = NeuralMonotoneTSL(K=1.0e4, hidden=16)
    optimizer = torch.optim.Adam(law.damage_net.parameters(), lr=0.03)
    for _ in range(3000):
        optimizer.zero_grad()
        pred = law.damage_net(kappas_train)
        loss = ((pred - targets) ** 2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_pred = law.damage_net(kappas_train)
        max_err = (final_pred - targets).abs().max().item()

    assert max_err < 0.03
    # Admissibility survives training (architectural, not a training-time penalty).
    assert torch.all(final_pred >= -1e-9)
    assert torch.all(final_pred <= 1.0 + 1e-9)
    assert torch.all(final_pred[1:] - final_pred[:-1] >= -1e-6)
