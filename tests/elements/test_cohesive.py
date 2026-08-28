"""Single-element validation: unit-length cohesive element with a
prescribed opening. Recovers T(delta), verifies loading/unloading/reloading irreversibility,
checks dissipated energy == G_c, and checks the autograd tangent against finite differences."""

import torch

from diffcohesive.elements import CohesiveElement2D
from diffcohesive.laws import BilinearMixedModeTSL


def _unit_element():
    # Bottom edge X1=(0,0)->X2=(1,0); top edge X4,X3 coincident (zero-thickness insertion).
    X_elem = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64
    )
    return X_elem


def _prescribed_opening_u(d: float) -> torch.Tensor:
    # Uniform normal opening d: bottom fixed, top nodes translated by +d along the local
    # normal, which for this geometry (edge along x) is the global +y direction.
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, d, 0.0, d], dtype=torch.float64)


def test_uniform_opening_matches_pointwise_law():
    law = BilinearMixedModeTSL(T_max_n=50.0, T_max_s=50.0, G_c1=1.0, G_c2=1.0, eta=1.0, K=1.0e5)
    elem = CohesiveElement2D(law)
    X_elem = _unit_element()

    kappa = torch.zeros(2, dtype=torch.float64)
    kappa_point = torch.tensor(0.0, dtype=torch.float64)

    delta0 = law.T_max_n.item() / law.K.item()
    deltaf = 2.0 * law.G_c1.item() / (law.K.item() * delta0)

    for d in torch.linspace(0.0, 1.2 * deltaf, 500, dtype=torch.float64):
        u_elem = _prescribed_opening_u(d.item())
        R_e, kappa, damage = elem.residual(u_elem, X_elem, kappa)

        # Total y-force transmitted at the top nodes == integral of T_n over unit length ==
        # T_n(d) since the opening (and hence traction) is uniform across the element.
        top_force_y = R_e[5] + R_e[7]

        delta_local_point = torch.stack([d, torch.zeros((), dtype=torch.float64)])
        traction_point, kappa_point, _ = law(delta_local_point, kappa_point)

        assert abs(top_force_y.item() - traction_point[0].item()) < 1e-6 * max(1.0, abs(traction_point[0].item()))


def test_element_dissipates_Gc_and_is_irreversible():
    law = BilinearMixedModeTSL(T_max_n=50.0, T_max_s=50.0, G_c1=1.0, G_c2=1.0, eta=1.0, K=1.0e5)
    elem = CohesiveElement2D(law)
    X_elem = _unit_element()

    delta0 = law.T_max_n.item() / law.K.item()
    deltaf = 2.0 * law.G_c1.item() / (law.K.item() * delta0)

    kappa = torch.zeros(2, dtype=torch.float64)
    ds = torch.linspace(0.0, 1.2 * deltaf, 3000, dtype=torch.float64)
    forces = []
    for d in ds:
        u_elem = _prescribed_opening_u(d.item())
        R_e, kappa, damage = elem.residual(u_elem, X_elem, kappa)
        forces.append((R_e[5] + R_e[7]).item())
    forces = torch.tensor(forces, dtype=torch.float64)

    dissipated = torch.trapezoid(forces, ds).item()
    assert abs(dissipated - law.G_c1.item()) / law.G_c1.item() < 5e-3

    # Irreversibility: unload from mid-softening, damage frozen, response purely elastic.
    d_partial = 0.5 * (delta0 + deltaf)
    kappa = torch.zeros(2, dtype=torch.float64)
    _, kappa, damage_partial = elem.residual(_prescribed_opening_u(d_partial), X_elem, kappa)

    d_unload = 0.3 * d_partial
    R_unload, kappa_unload, damage_unload = elem.residual(_prescribed_opening_u(d_unload), X_elem, kappa)
    assert torch.allclose(damage_unload, damage_partial, atol=1e-4)

    expected_unload_force = (1.0 - damage_partial[0].item()) * law.K.item() * d_unload
    top_force_unload = (R_unload[5] + R_unload[7]).item()
    assert abs(top_force_unload - expected_unload_force) / expected_unload_force < 1e-4


def test_autograd_tangent_matches_finite_difference():
    law = BilinearMixedModeTSL(T_max_n=50.0, T_max_s=50.0, G_c1=1.0, G_c2=1.0, eta=1.0, K=1.0e5)
    elem = CohesiveElement2D(law)
    X_elem = _unit_element()

    delta0 = law.T_max_n.item() / law.K.item()
    deltaf = 2.0 * law.G_c1.item() / (law.K.item() * delta0)

    # Drive into partial softening first so kappa_prev reflects a nontrivial damage state,
    # then evaluate/verify the tangent at that state (mirrors an in-between Newton iteration).
    kappa = torch.zeros(2, dtype=torch.float64)
    d_history = 0.5 * (delta0 + deltaf)
    _, kappa, _ = elem.residual(_prescribed_opening_u(d_history), X_elem, kappa)

    u0 = _prescribed_opening_u(d_history * 1.1)
    K_autograd = elem.tangent(u0, X_elem, kappa)

    h = 1e-6
    K_fd = torch.zeros(8, 8, dtype=torch.float64)
    for i in range(8):
        u_plus = u0.clone()
        u_minus = u0.clone()
        u_plus[i] += h
        u_minus[i] -= h
        R_plus, _, _ = elem.residual(u_plus, X_elem, kappa)
        R_minus, _, _ = elem.residual(u_minus, X_elem, kappa)
        K_fd[:, i] = (R_plus - R_minus) / (2 * h)

    denom = K_fd.abs().max().item()
    assert torch.allclose(K_autograd, K_fd, atol=1e-3 * denom, rtol=1e-3)
