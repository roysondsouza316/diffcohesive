"""Mesh-level extension of tests/laws/test_bilinear_mixed_mode.py's point-level BK checks
(the mesh-level stand-in for literal ENF/MMB contact fixtures): dissipated energy
recovered from a real assembled-mesh solve must match the Benzeggagh-Kenane mixed-mode
toughness prediction across a sweep of loading angles, not just at a single quadrature point."""

import pytest

from examples.mixed_mode_beam import DEFAULT_PARAMS, run_mixed_mode


@pytest.mark.parametrize("theta_deg", [0, 30, 45, 60, 90])
def test_mesh_level_dissipation_matches_bk_prediction(theta_deg):
    result = run_mixed_mode(theta_deg, **DEFAULT_PARAMS)
    rel_err = abs(result["dissipated"] - result["Gc_predicted"]) / result["Gc_predicted"]
    assert rel_err < 0.03
