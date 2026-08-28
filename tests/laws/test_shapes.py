"""Validation of the TSL shape library (laws/shapes.py) against Alfano, Compos. Sci. Technol.
66 (2006) 723-730: closed-form internal parameters must reproduce the paper's own computed
values (its Table 1 reports b1 and beta for both property sets), every shape must dissipate
exactly Gc with the same K0/sigma0 peak, and irreversibility must hold (the paper's laws are
holonomic -- ours add frozen-secant unloading, which must not change the monotonic envelope)."""

import pytest
import torch

from diffcohesive.laws import (
    BilinearShapeTSL,
    ExponentialTSL,
    LinearParabolicTSL,
    TrapezoidalTSL,
    SHAPE_LAWS,
)

# Paper Table 1: mode-I property set and pull-out property set, with the paper's own
# computed internal parameters for the linear-parabolic (b1) and exponential (beta) laws.
MODE_I_SET = dict(K0=10000.0, sigma0=30.0, Gc=0.5)
MODE_I_B1 = 0.0106725
MODE_I_BETA = 185.028

PULLOUT_SET = dict(K0=5000.0, sigma0=3.0, Gc=0.1)
PULLOUT_B1 = 0.019025
PULLOUT_BETA = 62.1855


def test_linear_parabolic_b1_matches_paper_computed_values():
    law = LinearParabolicTSL(**MODE_I_SET)
    assert abs(law.b1().item() - MODE_I_B1) / MODE_I_B1 < 1e-4
    law2 = LinearParabolicTSL(**PULLOUT_SET)
    assert abs(law2.b1().item() - PULLOUT_B1) / PULLOUT_B1 < 1e-4


def test_exponential_beta_matches_paper_computed_values():
    law = ExponentialTSL(**MODE_I_SET)
    assert abs(law.beta().item() - MODE_I_BETA) / MODE_I_BETA < 1e-4
    law2 = ExponentialTSL(**PULLOUT_SET)
    assert abs(law2.beta().item() - PULLOUT_BETA) / PULLOUT_BETA < 1e-4


def _monotonic_mode_i_dissipation(law, d_max, n_steps=8000):
    deltas = torch.linspace(0.0, d_max, n_steps, dtype=torch.float64)
    kappa = torch.tensor(0.0, dtype=torch.float64)
    tractions = []
    for d in deltas:
        delta_local = torch.stack([d, torch.zeros((), dtype=torch.float64)])
        traction, kappa, _ = law(delta_local, kappa)
        tractions.append(traction[0])
    tractions = torch.stack(tractions)
    return torch.trapezoid(tractions, deltas).item(), tractions


@pytest.mark.parametrize("name", list(SHAPE_LAWS.keys()))
def test_every_shape_dissipates_Gc_and_peaks_at_sigma0(name):
    params = MODE_I_SET
    law = SHAPE_LAWS[name](**params)
    # Integrate well past complete failure; the exponential tail needs the longest range.
    d_max = 6.0 * 2.0 * params["Gc"] / params["sigma0"]
    dissipated, tractions = _monotonic_mode_i_dissipation(law, d_max)
    assert abs(dissipated - params["Gc"]) / params["Gc"] < 1e-2
    assert abs(tractions.max().item() - params["sigma0"]) / params["sigma0"] < 1e-2
    # Fully failed at the end of the range.
    assert tractions[-1].item() < 1e-2 * params["sigma0"]


@pytest.mark.parametrize("name", ["linear-parabolic", "exponential", "trapezoidal"])
def test_unloading_follows_frozen_secant_and_damage_is_irreversible(name):
    params = MODE_I_SET
    law = SHAPE_LAWS[name](**params)
    kappa = torch.tensor(0.0, dtype=torch.float64)

    def step(d_val):
        nonlocal kappa
        delta_local = torch.tensor([d_val, 0.0], dtype=torch.float64)
        traction, kappa, damage = law(delta_local, kappa)
        return traction[0].item(), damage.item()

    # Load partway into softening (past the peak for every shape).
    d_soft = 3.0 * params["sigma0"] / params["K0"] + 0.3 * params["Gc"] / params["sigma0"]
    _, D_soft = step(d_soft)
    assert 0.0 < D_soft < 1.0

    # Unload: damage frozen, traction on the secant (1-D)*K0*delta.
    d_unl = 0.4 * d_soft
    T_unl, D_unl = step(d_unl)
    assert abs(D_unl - D_soft) < 1e-6
    expected = (1.0 - D_soft) * params["K0"] * d_unl
    assert abs(T_unl - expected) / abs(expected) < 1e-6

    # Push past the previous maximum: damage resumes growing.
    _, D_beyond = step(1.3 * d_soft)
    assert D_beyond > D_soft
