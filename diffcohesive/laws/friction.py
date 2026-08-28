"""Combined interface damage + friction, following Alfano & Sacco, "Combining interface
damage and friction in a cohesive-zone model", Int. J. Numer. Meth. Engng 68 (2006) 542-582,
doi:10.1002/nme.1728.

Mesomechanical idea (their Section 2): a representative elementary area of interface is split
into an undamaged fraction (1-D) that behaves elastically and a damaged fraction D on which
unilateral contact with Coulomb friction acts. The homogenized stress is

    sigma = (1-D) * K1 * s1 + D * sigma_d,   sigma_d = K1 * min(s1, 0)      (no-tension contact)
    tau   = (1-D) * K2 * s2 + D * tau_d,     tau_d   = K2 * (s2 - s2_slip)

with the inelastic slip s2_slip evolving by a non-associative Coulomb law (their Eqs. 9-11,
no dilatancy): slip occurs when |tau_d| + mu * sigma_d > 0, and the return map brings |tau_d|
back to -mu * sigma_d. Friction therefore appears gradually as damage grows -- zero on the
intact interface, full Coulomb once decohesion completes -- with no extra model ingredients.

Damage follows Crisfield's bilinear model in the damage-mechanics form used by the paper
(their Eqs. 14-17): with so_i = onset separations, sc_i = 2*Gc_i/peak_i final separations, and
the proportionality hypothesis so1/sc1 = so2/sc2 = 1 - eta,

    lambda = sqrt( (<s1>+ / so1)^2 + (s2 / so2)^2 ),  lambda_max = max over history,
    D = clamp( (1/eta) * (1 - 1/lambda_max), 0, 1 ).

For pure mode I tension this reduces exactly to the bilinear law (validated in tests). The
history state is two components per quadrature point (state_dim = 2): [lambda_max, s2_slip].

All non-smooth pieces (Macaulay brackets, |tau|, sign, history max) use the same smoothed
primitives as the rest of the package so the law stays autograd-friendly.
"""

from typing import Dict, Optional

import torch

from .base import TractionSeparationLaw
from .smoothing import smooth_macaulay, smooth_max


def _get(name, default, params):
    if params is not None and name in params:
        return params[name]
    return default


class FrictionalCohesiveTSL(TractionSeparationLaw):
    """Learnable params theta = {sigma0, tau0, G_c1, G_c2, K1, K2, mu}.

    Note the paper's hypothesis so1/sc1 = so2/sc2 relates the inputs; eta is derived from the
    mode-II set (eta = 1 - tau0^2 / (2*K2*G_c2)). Supplying a mode-I set with a different
    implied eta is allowed but slightly departs from the paper's single-eta damage measure --
    choose consistent parameters when strict fidelity matters (the tests do)."""

    state_dim = 2

    def __init__(
        self,
        sigma0: float,
        tau0: float,
        G_c1: float,
        G_c2: float,
        K1: float,
        K2: float,
        mu: float,
        smoothing_fraction: float = 1.0e-3,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.sigma0 = torch.nn.Parameter(torch.as_tensor(sigma0, dtype=dtype))
        self.tau0 = torch.nn.Parameter(torch.as_tensor(tau0, dtype=dtype))
        self.G_c1 = torch.nn.Parameter(torch.as_tensor(G_c1, dtype=dtype))
        self.G_c2 = torch.nn.Parameter(torch.as_tensor(G_c2, dtype=dtype))
        self.K1 = torch.nn.Parameter(torch.as_tensor(K1, dtype=dtype))
        self.K2 = torch.nn.Parameter(torch.as_tensor(K2, dtype=dtype))
        self.mu = torch.nn.Parameter(torch.as_tensor(mu, dtype=dtype))
        self.smoothing_fraction = smoothing_fraction

    def forward(
        self,
        delta_local: torch.Tensor,
        kappa_prev: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        sigma0 = _get("sigma0", self.sigma0, params)
        tau0 = _get("tau0", self.tau0, params)
        G_c1 = _get("G_c1", self.G_c1, params)
        G_c2 = _get("G_c2", self.G_c2, params)
        K1 = _get("K1", self.K1, params)
        K2 = _get("K2", self.K2, params)
        mu = _get("mu", self.mu, params)

        s1 = delta_local[..., 0]
        s2 = delta_local[..., 1]

        so1 = sigma0 / K1
        so2 = tau0 / K2
        sc2 = 2.0 * G_c2 / tau0
        eta = 1.0 - so2 / sc2

        lam_prev = kappa_prev[..., 0]
        slip_prev = kappa_prev[..., 1]

        eps = self.smoothing_fraction * so1
        eps_dimless = self.smoothing_fraction

        # -- Crisfield damage from the dimensionless effective measure lambda (Eqs. 15-17) --
        s1_plus = smooth_macaulay(s1, eps)
        lam = torch.sqrt((s1_plus / so1) ** 2 + (s2 / so2) ** 2 + 1.0e-24)
        lam_max = smooth_max(lam_prev, lam, eps_dimless)
        damage = torch.clamp((1.0 - 1.0 / lam_max.clamp_min(1.0e-12)) / eta, min=0.0, max=1.0)

        # -- Unilateral contact stress on the damaged fraction (Eq. 5): no tension --
        sigma_d = -K1 * smooth_macaulay(-s1, eps)

        # -- Coulomb friction return map on the damaged fraction (Eqs. 9-11, 13) --
        tau_d_trial = K2 * (s2 - slip_prev)
        eps_t = self.smoothing_fraction * tau0
        abs_trial = torch.sqrt(tau_d_trial ** 2 + eps_t ** 2)
        phi = abs_trial + mu * sigma_d  # sigma_d <= 0, so this is |tau| - mu*|sigma_contact|
        slip_increment = smooth_macaulay(phi, eps_t) / K2
        sign_trial = tau_d_trial / abs_trial
        slip_new = slip_prev + slip_increment * sign_trial
        tau_d = K2 * (s2 - slip_new)

        # -- Homogenized stress over the representative area (Eqs. 7, 12) --
        sigma = (1.0 - damage) * K1 * s1 + damage * sigma_d
        tau = (1.0 - damage) * K2 * s2 + damage * tau_d

        traction = torch.stack([sigma, tau], dim=-1)
        kappa_new = torch.stack([lam_max, slip_new], dim=-1)
        return traction, kappa_new, damage
