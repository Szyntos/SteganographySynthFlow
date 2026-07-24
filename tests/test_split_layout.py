import numpy as np
import pytest

from synthflow.Framing.SplitLayout import SplitLayout


def test_two_split_envelope_and_data_start():
    layout = SplitLayout.two_split(16)
    assert layout.phases == 2
    assert layout.data_start == 8
    assert np.all(layout.envelope[:8] == 0.0)
    assert np.all(layout.envelope[8:] == 1.0)


def test_four_split_requires_divisible_by_four():
    with pytest.raises(ValueError):
        SplitLayout.four_split(10)


def test_four_split_envelope_shape():
    layout = SplitLayout.four_split(16)
    q = 4
    assert layout.phases == 4
    assert layout.data_start == 2 * q

    assert np.all(layout.envelope[:q] == 0.0)

    ramp_up = layout.envelope[q:2 * q]
    assert np.allclose(ramp_up, np.arange(q) / q)

    assert np.all(layout.envelope[2 * q:3 * q] == 1.0)

    ramp_down = layout.envelope[3 * q:4 * q]
    assert np.allclose(ramp_down, 1.0 - np.arange(q) / q)


def test_four_split_degenerate_q_equals_one():
    layout = SplitLayout.four_split(4)
    assert np.allclose(layout.envelope, [0.0, 1.0, 1.0, 0.0])
