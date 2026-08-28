"""Neural (thermodynamically admissible) traction-separation law.

``T = (1 - D_NN(kappa; theta)) * K * delta``, the same continuum-damage form as the bilinear
law, but with the damage-vs-history mapping D_NN a learned *monotone* network instead of a
fixed analytic formula. The monotone-network construction guarantees D in [0,1] and dD/dkappa
>= 0 architecturally (not via a training-time penalty): non-negative weights composed through
monotone-increasing activations (softplus -> positive weights; tanh, sigmoid -> monotone), then
referenced against the network's own value at kappa=0 so D(0) = 0 exactly. Because the traction
retains the same (1-D)*K*delta form as the bilinear law, unloading again follows a frozen-D
secant to the origin, so dissipation is non-negative for the same reason -- non-negative
dissipation needs no separate penalty term needed for either constraint.
"""

from typing import Optional, Dict

import torch
import torch.nn.functional as F

from .base import TractionSeparationLaw
from .smoothing import smooth_macaulay, smooth_max


class MonotoneDamageNet(torch.nn.Module):
    def __init__(self, hidden: int = 8, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.w1_raw = torch.nn.Parameter(torch.randn(hidden, dtype=dtype) * 0.5)
        self.b1 = torch.nn.Parameter(torch.randn(hidden, dtype=dtype) * 0.5)
        self.w2_raw = torch.nn.Parameter(torch.randn(hidden, dtype=dtype) * 0.5)
        self.b2 = torch.nn.Parameter(torch.zeros((), dtype=dtype))

    def _logit(self, kappa: torch.Tensor) -> torch.Tensor:
        w1 = F.softplus(self.w1_raw)
        w2 = F.softplus(self.w2_raw)
        hidden = torch.tanh(kappa.unsqueeze(-1) * w1 + self.b1)
        return (hidden * w2).sum(-1) + self.b2

    def forward(self, kappa: torch.Tensor) -> torch.Tensor:
        f_k = self._logit(kappa)
        f_0 = self._logit(torch.zeros_like(kappa))
        return torch.sigmoid(f_k) - torch.sigmoid(f_0)


class NeuralMonotoneTSL(TractionSeparationLaw):
    def __init__(self, K: float, hidden: int = 8, smoothing_eps: float = 1.0e-4, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.K = torch.nn.Parameter(torch.as_tensor(K, dtype=dtype))
        self.damage_net = MonotoneDamageNet(hidden=hidden, dtype=dtype)
        self.smoothing_eps = smoothing_eps

    def forward(
        self,
        delta_local: torch.Tensor,
        kappa_prev: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        delta_n = delta_local[..., 0]
        shear = delta_local[..., 1:]
        eps_shear = 1.0e-12
        delta_s = torch.sqrt(shear.pow(2).sum(-1) + eps_shear)

        eps = self.smoothing_eps
        mn = smooth_macaulay(delta_n, eps)
        lam = torch.sqrt(mn * mn + delta_s * delta_s)
        kappa_new = smooth_max(kappa_prev, lam, eps)

        damage = self.damage_net(kappa_new)

        K = self.K
        traction_n = K * delta_n - damage * K * mn
        traction_s = (1.0 - damage).unsqueeze(-1) * K * shear
        traction = torch.cat([traction_n.unsqueeze(-1), traction_s], dim=-1)
        return traction, kappa_new, damage
