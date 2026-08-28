"""Interface-law shape library, following Alfano, "On the influence of the shape of the
interface law on the application of cohesive-zone models", Composites Science and Technology
66 (2006) 723-730, doi:10.1016/j.compscitech.2004.12.024.

Four pure-mode envelope shapes -- bilinear, linear-parabolic, exponential, trapezoidal -- all
parametrized by the SAME three inputs (initial stiffness K0, peak traction sigma0, fracture
energy Gc), so that differences in a simulation are attributable purely to the shape (the
paper's premise). Internal shape parameters are closed-form (derived below), and two of them
are cross-checked against the paper's own computed values in tests/laws/test_shapes.py:
b1 = 0.0106725 mm (linear-parabolic, mode-I property set) and beta = 185.028 1/mm
(exponential, same set).

Each shape is wrapped in the same damage/history formalism as laws/bilinear.py so the laws are
irreversible (frozen-secant unloading) rather than the paper's holonomic simplification:
    lambda = sqrt(<delta_n>^2 + delta_s^2),  kappa = max_history(lambda)
    D(kappa) = clamp(1 - sigma_env(kappa) / (K0*kappa), 0, 1)
    T_n = K0*delta_n - D*K0*<delta_n>   (compression keeps full penalty stiffness)
    T_s = (1-D)*K0*delta_s
For monotonic loading this reproduces the paper's envelope exactly; the secant-damage form
guarantees D monotone because every envelope's secant sigma_env(d)/d is non-increasing.

Unlike BilinearMixedModeTSL these laws use one (K0, sigma0, Gc) set for all modes (exactly the
paper's pure-mode setting) and have no BK/power-law mixed-mode toughness closure -- use
BilinearMixedModeTSL for genuinely mixed-mode problems.
"""

import math
from typing import Dict, Optional

import torch

from .base import TractionSeparationLaw
from .smoothing import smooth_macaulay, smooth_max

# Exact coefficient of the linear-parabolic law's softening-branch energy (integral of
# 1/2 + x - x^2/2 from 0 to its positive root x* = 1 + sqrt(2)): (5 + 4*sqrt(2)) / 6.
_LP_ENERGY_COEFF = (5.0 + 4.0 * math.sqrt(2.0)) / 6.0
_LP_ROOT = 1.0 + math.sqrt(2.0)


class _ShapedTSL(TractionSeparationLaw):
    """Common irreversible-damage wrapper; subclasses provide the traction envelope."""

    def __init__(self, K0: float, sigma0: float, Gc: float,
                 smoothing_fraction: float = 1.0e-3, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.K0 = torch.nn.Parameter(torch.as_tensor(K0, dtype=dtype))
        self.sigma0 = torch.nn.Parameter(torch.as_tensor(sigma0, dtype=dtype))
        self.Gc = torch.nn.Parameter(torch.as_tensor(Gc, dtype=dtype))
        self.smoothing_fraction = smoothing_fraction

    def envelope(self, delta: torch.Tensor, K0, sigma0, Gc) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, delta_local: torch.Tensor, kappa_prev: torch.Tensor,
                params: Optional[Dict[str, torch.Tensor]] = None):
        K0 = params["K0"] if params is not None and "K0" in params else self.K0
        sigma0 = params["sigma0"] if params is not None and "sigma0" in params else self.sigma0
        Gc = params["Gc"] if params is not None and "Gc" in params else self.Gc

        delta_n = delta_local[..., 0]
        shear = delta_local[..., 1:]
        eps_shear = 1.0e-12
        delta_s = torch.sqrt(shear.pow(2).sum(-1) + eps_shear)

        eps = self.smoothing_fraction * sigma0 / K0
        mn = smooth_macaulay(delta_n, eps)
        lam = torch.sqrt(mn * mn + delta_s * delta_s)
        kappa_new = smooth_max(kappa_prev, lam, eps)

        kappa_safe = kappa_new.clamp_min(1.0e-15)
        secant = self.envelope(kappa_safe, K0, sigma0, Gc) / (K0 * kappa_safe)
        damage = torch.clamp(1.0 - secant, min=0.0, max=1.0)

        traction_n = K0 * delta_n - damage * K0 * mn
        traction_s = (1.0 - damage).unsqueeze(-1) * K0 * shear
        traction = torch.cat([traction_n.unsqueeze(-1), traction_s], dim=-1)
        return traction, kappa_new, damage


class BilinearShapeTSL(_ShapedTSL):
    """Paper Eq. (1)-(2): a0 = sigma0/K0, a1 = 2*Gc/sigma0 (single-mode bilinear shape)."""

    def envelope(self, delta, K0, sigma0, Gc):
        a0 = sigma0 / K0
        a1 = 2.0 * Gc / sigma0
        rising = K0 * delta
        softening = sigma0 * (a1 - delta) / (a1 - a0)
        env = torch.where(delta <= a0, rising, softening)
        return env.clamp_min(0.0)


class LinearParabolicTSL(_ShapedTSL):
    """Paper Eq. (3)-(5): linear to sigma0/2 at b0 = sigma0/(2*K0), then a parabola peaking at
    sigma0 at delta = b1 and falling to zero at b0 + (1+sqrt(2))*(b1-b0).

    Energy closure (paper's Eq. (5), here in closed form): the softening branch integrates to
    _LP_ENERGY_COEFF * sigma0 * (b1-b0), so
        b1 = b0 + (Gc - sigma0^2/(8*K0)) / (_LP_ENERGY_COEFF * sigma0).
    """

    def b1(self, K0=None, sigma0=None, Gc=None):
        K0 = self.K0 if K0 is None else K0
        sigma0 = self.sigma0 if sigma0 is None else sigma0
        Gc = self.Gc if Gc is None else Gc
        b0 = sigma0 / (2.0 * K0)
        return b0 + (Gc - sigma0 ** 2 / (8.0 * K0)) / (_LP_ENERGY_COEFF * sigma0)

    def envelope(self, delta, K0, sigma0, Gc):
        b0 = sigma0 / (2.0 * K0)
        b1 = self.b1(K0, sigma0, Gc)
        x = (delta - b0) / (b1 - b0)
        parabola = sigma0 * (0.5 + x - 0.5 * x * x)
        env = torch.where(delta <= b0, K0 * delta, parabola.clamp_min(0.0))
        return env


class ExponentialTSL(_ShapedTSL):
    """Paper Eq. (6)-(8): sigma = K0*delta*exp(-delta/c0) up to c0 = sigma0*e/K0 (peak sigma0),
    then sigma0*(1 + beta*(delta-c0))*exp(-beta*(delta-c0)).

    Energy closure (paper's Eq. (8), closed form): the rising part integrates to
    K0*c0^2*(1 - 2/e) and the tail to 2*sigma0/beta, so
        beta = 2*sigma0 / (Gc - K0*c0^2*(1 - 2/e)).
    """

    def beta(self, K0=None, sigma0=None, Gc=None):
        K0 = self.K0 if K0 is None else K0
        sigma0 = self.sigma0 if sigma0 is None else sigma0
        Gc = self.Gc if Gc is None else Gc
        c0 = sigma0 * math.e / K0
        rising_energy = K0 * c0 ** 2 * (1.0 - 2.0 / math.e)
        return 2.0 * sigma0 / (Gc - rising_energy)

    def envelope(self, delta, K0, sigma0, Gc):
        c0 = sigma0 * math.e / K0
        beta = self.beta(K0, sigma0, Gc)
        rising = K0 * delta * torch.exp(-delta / c0)
        x = beta * (delta - c0)
        tail = sigma0 * (1.0 + x) * torch.exp(-x)
        return torch.where(delta <= c0, rising, tail)


class TrapezoidalTSL(_ShapedTSL):
    """Paper Eq. (9)-(10) (Tvergaard-Hutchinson type): linear to sigma0 at d0 = sigma0/K0,
    plateau to d1 = Gc/sigma0, linear down to zero at d2 = d0 + d1."""

    def envelope(self, delta, K0, sigma0, Gc):
        d0 = sigma0 / K0
        d1 = Gc / sigma0
        d2 = d0 + d1
        env = torch.where(
            delta <= d0,
            K0 * delta,
            torch.where(
                delta <= d1,
                sigma0 * torch.ones_like(delta),
                (sigma0 * (d2 - delta) / (d2 - d1)).clamp_min(0.0),
            ),
        )
        return env


class SmithFerranteTSL(_ShapedTSL):
    """The Smith-Ferrante exponential law as used, e.g., in the COMET-FEniCSx intrinsic-CZM
    tour (Bleyer, https://bleyerj.github.io/comet-fenicsx/tours/interfaces/intrinsic_czm/):

        T = (G_c / delta_0^2) * exp(-delta/delta_0) * [[u]],   delta_0 = G_c / (sigma_c * e),

    with effective opening delta = sqrt(<delta_n>^2 + beta*delta_s^2) and damage
    d = 1 - exp(-max(delta)/delta_0). Only two parameters are independent (G_c, sigma_c);
    the initial stiffness is K_0 = G_c/delta_0^2 and the envelope integrates to exactly G_c.
    The damage variable of that reference IS this package's secant damage for this envelope:
    D(kappa) = 1 - sigma_env(kappa)/(K_0*kappa) = 1 - exp(-kappa/delta_0) -- verified in
    tests/laws/test_smith_ferrante.py, together with the normalized traction-opening curve
    T/sigma_c = (delta/delta_0) * exp(1 - delta/delta_0).

    Documented difference from the reference: normal compression here meets the full penalty
    stiffness K_0 (no interpenetration, Macaulay-bracketed normal opening), whereas the
    reference degrades compression like tension."""

    def __init__(self, Gc: float, sigma_c: float, beta: float = 1.0,
                 smoothing_fraction: float = 1.0e-3, dtype: torch.dtype = torch.float64):
        delta_0 = Gc / (sigma_c * math.e)
        K0 = Gc / delta_0 ** 2
        super().__init__(K0=K0, sigma0=sigma_c, Gc=Gc,
                         smoothing_fraction=smoothing_fraction, dtype=dtype)
        self.beta = beta

    def delta_0(self, K0=None, sigma0=None):
        sigma0 = self.sigma0 if sigma0 is None else sigma0
        K0 = self.K0 if K0 is None else K0
        # delta_0 = Gc/(sigma_c*e) expressed through the stored (K0, sigma0): peak at delta_0
        # where K0*delta_0*exp(-1) = sigma0.
        return sigma0 * math.e / K0

    def envelope(self, delta, K0, sigma0, Gc):
        d0 = self.delta_0(K0, sigma0)
        return K0 * delta * torch.exp(-delta / d0)

    def forward(self, delta_local, kappa_prev, params=None):
        # The reference's beta enters ONLY the damage-driving effective opening,
        # delta = sqrt(<dn>^2 + beta*ds^2); the traction map stays isotropic,
        # T = (1-d) K0 [[u]] (with our Macaulay'd compression penalty on the normal part).
        # An earlier implementation scaled the tangential STIFFNESS by beta as well -- caught
        # by the cross-code comparison against the FEniCSx reference (30% peak overshoot on
        # the two-inclusion problem) and corrected here.
        K0 = params["K0"] if params is not None and "K0" in params else self.K0
        sigma0 = params["sigma0"] if params is not None and "sigma0" in params else self.sigma0
        Gc = params["Gc"] if params is not None and "Gc" in params else self.Gc

        delta_n = delta_local[..., 0]
        shear = delta_local[..., 1:]
        eps = self.smoothing_fraction * sigma0 / K0
        mn = smooth_macaulay(delta_n, eps)
        lam = torch.sqrt(mn * mn + self.beta * shear.pow(2).sum(-1) + 1.0e-24)
        kappa_new = smooth_max(kappa_prev, lam, eps)

        kappa_safe = kappa_new.clamp_min(1.0e-15)
        secant = self.envelope(kappa_safe, K0, sigma0, Gc) / (K0 * kappa_safe)
        damage = torch.clamp(1.0 - secant, min=0.0, max=1.0)

        traction_n = K0 * delta_n - damage * K0 * mn
        traction_s = (1.0 - damage).unsqueeze(-1) * K0 * shear
        traction = torch.cat([traction_n.unsqueeze(-1), traction_s], dim=-1)
        return traction, kappa_new, damage


SHAPE_LAWS = {
    "bilinear": BilinearShapeTSL,
    "linear-parabolic": LinearParabolicTSL,
    "exponential": ExponentialTSL,
    "trapezoidal": TrapezoidalTSL,
}
