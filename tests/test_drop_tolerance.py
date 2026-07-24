from synthflow.core.DropTolerance import DropAction, DropTolerance, DropToleranceConfig


def test_normal_when_not_missing():
    dt = DropTolerance(DropToleranceConfig(tolerance_chunks=3))
    assert dt.push(False) == DropAction.NORMAL
    assert dt.drop_run == 0


def test_tolerates_up_to_limit_then_resets_once():
    dt = DropTolerance(DropToleranceConfig(tolerance_chunks=3))
    assert dt.push(True) == DropAction.TOLERATE_MOCK
    assert dt.push(True) == DropAction.TOLERATE_MOCK
    assert dt.push(True) == DropAction.TOLERATE_MOCK
    assert dt.push(True) == DropAction.RESET_NOW
    assert dt.push(True) == DropAction.TOLERATE_MOCK


def test_missing_run_reset_by_normal_chunk():
    dt = DropTolerance(DropToleranceConfig(tolerance_chunks=2))
    dt.push(True)
    dt.push(True)
    assert dt.push(False) == DropAction.NORMAL
    assert dt.drop_run == 0
    assert dt.push(True) == DropAction.TOLERATE_MOCK


def test_zero_tolerance_resets_immediately():
    dt = DropTolerance(DropToleranceConfig(tolerance_chunks=0))
    assert dt.push(True) == DropAction.RESET_NOW
    assert dt.push(True) == DropAction.TOLERATE_MOCK


def test_reset_clears_drop_run_mid_run():
    dt = DropTolerance(DropToleranceConfig(tolerance_chunks=3))
    dt.push(True)
    dt.push(True)
    dt.reset()
    assert dt.drop_run == 0
    assert dt.push(True) == DropAction.TOLERATE_MOCK


def test_default_config_tolerance_is_three():
    dt = DropTolerance()
    assert dt.tolerance == 3
