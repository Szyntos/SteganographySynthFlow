import math

import numpy as np

from synthflow.core.EnergyGate import EnergyGate, EnergyGateConfig


def test_rms_of_empty_is_zero():
    assert EnergyGate.rms(np.array([])) == 0.0


def test_rms_matches_known_value():
    x = np.array([1.0, -1.0, 1.0, -1.0])
    assert math.isclose(EnergyGate.rms(x), 1.0, rel_tol=1e-9)


def test_first_call_seeds_ema_and_only_drops_below_abs_floor():
    gate = EnergyGate(EnergyGateConfig(abs_floor=0.1))
    assert gate.is_drop(0.5) is False
    assert gate.ema_valid is True
    assert gate.ema == 0.5

    gate2 = EnergyGate(EnergyGateConfig(abs_floor=0.1))
    assert gate2.is_drop(0.01) is True


def test_steady_state_drop_on_large_relative_dip():
    gate = EnergyGate(EnergyGateConfig(ema_alpha=0.5, abs_floor=1e-6, drop_ratio=0.25))
    gate.is_drop(1.0)
    for _ in range(20):
        gate.is_drop(1.0)
    assert gate.is_drop(0.01) is True


def test_steady_state_no_drop_when_near_ema():
    gate = EnergyGate(EnergyGateConfig(ema_alpha=0.5, abs_floor=1e-6, drop_ratio=0.25))
    for _ in range(20):
        gate.is_drop(1.0)
    assert gate.is_drop(0.95) is False


def test_reset_returns_to_invalid_state():
    gate = EnergyGate()
    gate.is_drop(1.0)
    assert gate.ema_valid is True
    gate.reset()
    assert gate.ema_valid is False
    assert gate.ema == 0.0
