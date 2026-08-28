"""Pluggable traction-separation law interface."""

from typing import Optional, Dict

import torch
import torch.nn as nn


class TractionSeparationLaw(nn.Module):
    """Maps a local separation + internal history state to a traction and an updated state.

    Subclasses implement either an analytic law (bilinear, exponential) or a neural
    network; both are calibrated by the same autograd-driven identification loop
    since both are ordinary ``nn.Module``s.
    """

    #: Number of internal history components per quadrature point. 1 for pure-damage laws
    #: (the history is the scalar max effective separation, shape (...,)); >1 for laws with
    #: extra internal variables (e.g. the frictional law carries [damage history, plastic
    #: slip], shape (..., state_dim)). CohesiveMeshModel.init_history sizes the global state
    #: accordingly; everything between (elements, solvers, implicit diff) treats the state as
    #: an opaque tensor.
    state_dim: int = 1

    def forward(
        self,
        delta_local: torch.Tensor,
        kappa_prev: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            delta_local: (..., ndim) local separations, ndim = 1 (normal) + shear components.
                Convention: delta_local[..., 0] is the normal separation, the remaining
                components are the (one or two) shear separations.
            kappa_prev: (...,) history variable for ``state_dim == 1`` laws (max effective
                separation reached so far), or (..., state_dim) for laws with additional
                internal variables.
            params: optional dict overriding this module's own parameters (e.g. for
                GPU-batched calibration over multiple parameter sets at once).

        Returns:
            traction: (..., ndim) traction in the local (normal/shear) frame.
            kappa_new: same shape as kappa_prev; the damage-history component is monotone
                non-decreasing.
            damage: (...,) scalar damage in [0, 1], monotone non-decreasing in the history.
        """
        raise NotImplementedError
