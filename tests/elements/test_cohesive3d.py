"""3D 8-node cohesive element validation, mirroring the 2D element's gates: uniform opening of a unit face must reproduce the pointwise law exactly, integrate to
G_c, respect irreversibility, resolve pure shear in both tangential directions, and the
autograd tangent must match finite differences."""

import torch

from diffcohesive.elements import CohesiveElement3D
from diffcohesive.laws import BilinearMixedModeTSL


def _unit_face_element(K=1.0e5, T_max=50.0, G_c=1.0):
    law = BilinearMixedModeTSL(T_max_n=T_max, T_max_s=T_max, G_c1=G_c, G_c2=G_c, eta=1.0, K=K)
    elem = CohesiveElement3D(law)
    # Unit square face in the x-z plane (normal = -y or +y depending on order); top nodes
    # coincident. Cyclic bottom order chosen so n = t_xi x t_eta = +y.
    X = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    return elem, X, law


def _uniform_opening_u(w):
    """All four top nodes displaced by +w in y (the face normal direction)."""
    u = torch.zeros(24, dtype=torch.float64)
    for i in range(4, 8):
        u[3 * i + 1] = w
    return u


def test_uniform_opening_reproduces_pointwise_law_and_dissipates_Gc():
    elem, X, law = _unit_face_element()
    K = law.K.item()
    delta0 = law.T_max_n.item() / K
    deltaf = 2.0 * law.G_c1.item() / (K * delta0)

    kappa = torch.zeros(elem.n_quad, dtype=torch.float64)
    openings = torch.linspace(0.0, 1.2 * deltaf, 3000, dtype=torch.float64)
    forces = []
    for w in openings:
        R_e, kappa, damage = elem.residual(_uniform_opening_u(w.item()), X, kappa)
        # Total normal force on the top face = sum of top-node y-components.
        forces.append(sum(R_e[3 * i + 1].item() for i in range(4, 8)))
    forces = torch.tensor(forces, dtype=torch.float64)

    # Unit face area: force == pointwise traction. Peak = T_max, energy = G_c.
    assert abs(forces.max().item() - law.T_max_n.item()) / law.T_max_n.item() < 1e-2
    dissipated = torch.trapezoid(forces, openings).item()
    assert abs(dissipated - law.G_c1.item()) / law.G_c1.item() < 1e-2
    assert abs(forces[-1].item()) < 1e-2 * law.T_max_n.item()
    assert torch.all(damage > 0.999)


def test_pure_shear_engages_shear_traction_in_both_tangent_directions():
    elem, X, law = _unit_face_element()
    K = law.K.item()
    for comp in (0, 2):  # global x and z are the two tangential directions of this face
        kappa = torch.zeros(elem.n_quad, dtype=torch.float64)
        u = torch.zeros(24, dtype=torch.float64)
        s = 1e-4  # elastic range
        for i in range(4, 8):
            u[3 * i + comp] = s
        R_e, _, damage = elem.residual(u, X, kappa)
        F_shear = sum(R_e[3 * i + comp].item() for i in range(4, 8))
        assert abs(F_shear - K * s) / (K * s) < 1e-6  # elastic: T = K*delta_s over unit area
        assert damage.max().item() < 1e-9


def test_autograd_tangent_matches_finite_differences():
    elem, X, law = _unit_face_element()
    kappa = torch.full((elem.n_quad,), 0.0, dtype=torch.float64)
    torch.manual_seed(0)
    u = 1e-4 * torch.randn(24, dtype=torch.float64)

    K_ad = elem.tangent(u, X, kappa)
    h = 1e-7
    K_fd = torch.zeros(24, 24, dtype=torch.float64)
    for j in range(24):
        up, um = u.clone(), u.clone()
        up[j] += h
        um[j] -= h
        Rp, _, _ = elem.residual(up, X, kappa)
        Rm, _, _ = elem.residual(um, X, kappa)
        K_fd[:, j] = (Rp - Rm) / (2 * h)
    assert (K_ad - K_fd).abs().max().item() < 1e-4 * K_fd.abs().max().item()
