import pytest

from synthflow.F0Estimator.F0Tracker import F0Tracker
from synthflow.core.Settings import Settings


def make_tracker():
    return F0Tracker(Settings())


def test_manual_mode_resolves_immediately_with_full_confidence():
    tracker = make_tracker()
    f0 = tracker.resolve([0.0] * 16, fs=8000.0)
    assert f0 == pytest.approx(400.0)
    assert tracker.confidence == 1.0
    assert tracker.has_pitch() is True


def test_set_manual_f0_defer_windows_delays_adoption():
    tracker = make_tracker()
    tracker.set_manual_f0(500.0, defer_windows=2)

    assert tracker.resolve([], fs=8000.0) == pytest.approx(400.0)
    assert tracker.resolve([], fs=8000.0) == pytest.approx(400.0)
    assert tracker.resolve([], fs=8000.0) == pytest.approx(500.0)


def test_set_manual_f0_no_defer_applies_next_resolve():
    tracker = make_tracker()
    tracker.set_manual_f0(500.0)
    assert tracker.resolve([], fs=8000.0) == pytest.approx(500.0)


def test_dirty_window_holds_last_value_without_touching_estimators():
    tracker = make_tracker()
    tracker.resolve([], fs=8000.0)
    held = tracker.f0
    assert tracker.resolve([], fs=8000.0, dirty=True) == held


def test_set_mode_invalid_raises():
    tracker = make_tracker()
    with pytest.raises(ValueError):
        tracker.set_mode("bogus")


def test_set_mode_resets_held_f0():
    tracker = make_tracker()
    tracker.resolve([], fs=8000.0)
    assert tracker.f0 > 0.0
    tracker.set_mode("autocorr")
    assert tracker.f0 == 0.0


def test_reset_clears_held_f0_and_confidence():
    tracker = make_tracker()
    tracker.resolve([], fs=8000.0)
    tracker.reset()
    assert tracker.f0 == 0.0
    assert tracker.confidence == 0.0
