"""Analytic bilinear mixed-mode cohesive law.

Camanho-Davila / Alfano-Crisfield bilinear traction-separation with the mixed-mode
onset/final-displacement closure of Turon et al. (2006), which reduces exactly to the pure
mode I / mode II bilinear laws in the single-mode limits and assumes equal penalty stiffness
K for normal and shear. Two mixed-mode fracture-toughness criteria are available via
``mixed_mode_criterion``:

- ``"bk"`` (default): Benzeggagh-Kenane,  G_c(B) = G_c1 + (G_c2 - G_c1) * B^eta.
- ``"power"``: the power-law criterion (G_I/G_c1)^alpha + (G_II/G_c2)^alpha = 1, which for a
  fixed mode ratio B = G_II/(G_I+G_II) closes to
  G_c(B) = [ ((1-B)/G_c1)^alpha + (B/G_c2)^alpha ]^(-1/alpha).

Both criteria reduce to G_c1 / G_c2 in the pure-mode limits. The single learnable exponent
parameter ``eta`` serves as the BK exponent or the power-law exponent alpha, depending on the
criterion selected -- this keeps the flat theta vector (theta_from_law) identical across
criteria, so the same calibration machinery applies unchanged.
"""

from typing import Optional, Dict

import torch

from .base import TractionSeparationLaw
from .smoothing import smooth_macaulay, smooth_max


def _get(name: str, default: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]):
    if params is not None and name in params:
        return params[name]
    return default


class BilinearMixedModeTSL(TractionSeparationLaw):
    """Learnable params theta = {T_max_n, T_max_s, G_c1, G_c2, eta, K}."""

    def __init__(
        self,
        T_max_n: float,
        T_max_s: float,
        G_c1: float,
        G_c2: float,
        eta: float = 1.0,
        K: float = 1.0e7,
        smoothing_fraction: float = 1.0e-3,
        mixed_mode_criterion: str = "bk",
        viscosity: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if mixed_mode_criterion not in ("bk", "power"):
            raise ValueError(f"mixed_mode_criterion must be 'bk' or 'power', got {mixed_mode_criterion!r}")
        self.T_max_n = torch.nn.Parameter(torch.as_tensor(T_max_n, dtype=dtype))
        self.T_max_s = torch.nn.Parameter(torch.as_tensor(T_max_s, dtype=dtype))
        self.G_c1 = torch.nn.Parameter(torch.as_tensor(G_c1, dtype=dtype))
        self.G_c2 = torch.nn.Parameter(torch.as_tensor(G_c2, dtype=dtype))
        self.eta = torch.nn.Parameter(torch.as_tensor(eta, dtype=dtype))
        self.K = torch.nn.Parameter(torch.as_tensor(K, dtype=dtype))
        # Numerical (non-learnable) transition width for the smoothed Macaulay/history-max,
        # scaled relative to the mode-I onset displacement each call.
        self.smoothing_fraction = smoothing_fraction
        # Structural (non-learnable) choice of mixed-mode toughness criterion; the learnable
        # exponent `eta` is the BK exponent or the power-law alpha accordingly.
        self.mixed_mode_criterion = mixed_mode_criterion
        # Optional viscous regularization of the damage variable (the analogue of Abaqus's
        # *DAMAGE STABILIZATION): the damage that degrades the traction is the Duvaut-Lions
        # relaxed variable D_v, updated once per load step (pseudo-time dt = 1) as
        #     D_v_new = (mu * D_v_prev + D_target) / (mu + 1),
        # so D_v lags the instantaneous bilinear damage D_target with relaxation constant
        # mu = ``viscosity`` (in load-step units; 0 disables and recovers the rate-independent
        # law exactly). This regularizes sharp element pop-ins under displacement control at
        # the cost of a small amount of artificial toughness -- keep mu small relative to the
        # number of steps over the softening branch. Non-learnable; with viscosity > 0 the
        # history state per quadrature point becomes [kappa, D_v] (state_dim = 2).
        self.viscosity = float(viscosity)
        self.state_dim = 2 if viscosity > 0.0 else 1

    def forward(
        self,
        delta_local: torch.Tensor,
        kappa_prev: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        T_max_n = _get("T_max_n", self.T_max_n, params)
        T_max_s = _get("T_max_s", self.T_max_s, params)
        G_c1 = _get("G_c1", self.G_c1, params)
        G_c2 = _get("G_c2", self.G_c2, params)
        eta = _get("eta", self.eta, params)
        K = _get("K", self.K, params)

        if self.state_dim == 2:
            damage_v_prev = kappa_prev[..., 1]
            kappa_prev = kappa_prev[..., 0]

        delta_n = delta_local[..., 0]
        shear = delta_local[..., 1:]
        # eps under the sqrt keeps delta_s differentiable at shear == 0 (safe-norm pattern).
        eps_shear = 1.0e-12
        delta_s = torch.sqrt(shear.pow(2).sum(-1) + eps_shear)

        delta0_n = T_max_n / K
        delta0_s = T_max_s / K
        eps = self.smoothing_fraction * delta0_n

        mn = smooth_macaulay(delta_n, eps)
        lam = torch.sqrt(mn * mn + delta_s * delta_s)

        # Displacement-based mixed-mode ratio (reduces to 0/1 in the pure normal/shear limits).
        mode_mix = delta_s * delta_s / (delta_s * delta_s + mn * mn + eps_shear)

        delta0_m = torch.sqrt(delta0_n * delta0_n + (delta0_s * delta0_s - delta0_n * delta0_n) * mode_mix)
        if self.mixed_mode_criterion == "power":
            # (G_I/G_c1)^alpha + (G_II/G_c2)^alpha = 1 at fixed mode ratio B = mode_mix; the
            # small clamp keeps the fractional power differentiable at the pure-mode limits.
            eps_pow = 1.0e-12
            B = mode_mix.clamp(eps_pow, 1.0 - eps_pow)
            G_c_m = (((1.0 - B) / G_c1).pow(eta) + (B / G_c2).pow(eta)).pow(-1.0 / eta)
        else:  # "bk"
            G_c_m = G_c1 + (G_c2 - G_c1) * mode_mix.pow(eta)
        deltaf_m = 2.0 * G_c_m / (K * delta0_m)

        kappa_new = smooth_max(kappa_prev, lam, eps)

        raw_damage = deltaf_m * (kappa_new - delta0_m) / (
            kappa_new.clamp_min(eps) * (deltaf_m - delta0_m).clamp_min(eps)
        )
        damage = torch.clamp(raw_damage, min=0.0, max=1.0)

        if self.state_dim == 2:
            mu = self.viscosity
            damage = (mu * damage_v_prev + damage) / (mu + 1.0)
            state_new = torch.stack([kappa_new, damage], dim=-1)
        else:
            state_new = kappa_new

        traction_n = K * delta_n - damage * K * mn
        traction_s = (1.0 - damage).unsqueeze(-1) * K * shear
        traction = torch.cat([traction_n.unsqueeze(-1), traction_s], dim=-1)

        return traction, state_new, damage
