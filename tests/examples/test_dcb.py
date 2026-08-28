"""Regression test for the DCB mesh-level validation: pre-peak compliance must
match classical DCB beam theory, and the response must show a genuine peak followed by
softening (not just monotone-elastic growth) -- confirms the cohesive zone is engaging at the
crack tip, not merely wiring the bulk elasticity."""

from examples.dcb import lefm_compliance, run_dcb


def test_dcb_prepeak_compliance_matches_lefm():
    result = run_dcb(
        length=15.0, arm_height=1.0, crack_length=5.0, nx=30, ny=4,
        n_disp_steps=25, max_disp=0.3, arc_steps=0,
    )
    delta, P = result["delta"], result["P"]
    compliance = lefm_compliance(result["mesh"].crack_length, 1000.0, 1.0)

    # Well before any damage onset (elastic regime). Simple beam theory neglects shear
    # deformation, so a few-percent deviation from the FEM compliance is expected, not a bug.
    check_idx = 5
    fem_compliance = delta[check_idx] / P[check_idx]
    assert abs(fem_compliance - compliance) / compliance < 0.08


def test_dcb_shows_peak_and_softening():
    result = run_dcb(
        length=15.0, arm_height=1.0, crack_length=5.0, nx=30, ny=4,
        n_disp_steps=60, max_disp=0.6, arc_ds=0.003, arc_steps=60,
    )
    P = result["P"]
    peak_idx = max(range(len(P)), key=lambda i: P[i])
    assert 0 < peak_idx < len(P) - 1  # interior peak: rose, then softened
    assert P[-1] < P[peak_idx]  # genuinely softened past the peak
