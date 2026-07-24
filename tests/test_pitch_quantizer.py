import math

from synthflow.F0Estimator.PitchQuantizer import quantize_to_chromatic_hz


def test_exact_a4_returns_440():
    assert quantize_to_chromatic_hz(440.0) == 440.0


def test_snaps_near_semitone_to_exact_hz():
    a5 = 440.0 * 2.0
    assert math.isclose(quantize_to_chromatic_hz(875.0), a5, rel_tol=1e-9)


def test_invalid_inputs_return_zero():
    assert quantize_to_chromatic_hz(0.0) == 0.0
    assert quantize_to_chromatic_hz(-100.0) == 0.0
    assert quantize_to_chromatic_hz(float("nan")) == 0.0
    assert quantize_to_chromatic_hz(float("inf")) == 0.0
    assert quantize_to_chromatic_hz(440.0, a4_hz=0.0) == 0.0


def test_custom_a4_reference_shifts_grid():
    assert quantize_to_chromatic_hz(432.0, a4_hz=432.0) == 432.0
