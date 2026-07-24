import pytest

from synthflow.io_devices.SamplesFifo import SamplesFifo


def test_no_threshold_starts_immediately():
    fifo = SamplesFifo()
    assert fifo.has_started() is True
    assert fifo.can_read(0) is True


def test_startup_threshold_blocks_until_reached():
    fifo = SamplesFifo(startup_threshold=5)
    assert fifo.has_started() is False
    fifo.push([1.0, 2.0, 3.0])
    assert fifo.has_started() is False
    assert fifo.can_read(1) is False
    fifo.push([4.0, 5.0])
    assert fifo.has_started() is True
    assert fifo.can_read(3) is True


def test_pop_raises_when_insufficient():
    fifo = SamplesFifo()
    fifo.push([1.0, 2.0])
    with pytest.raises(RuntimeError):
        fifo.pop(5)


def test_pop_or_empty_and_pop_or_silence_fallbacks():
    fifo = SamplesFifo(startup_threshold=3)
    fifo.push([1.0])
    assert fifo.pop_or_empty(2) == []
    assert fifo.pop_or_silence(2) == [0.0, 0.0]

    fifo.push([2.0, 3.0])
    assert fifo.pop_or_empty(2) == [1.0, 2.0]


def test_pop_consumes_fifo_order():
    fifo = SamplesFifo()
    fifo.push([1.0, 2.0, 3.0])
    assert fifo.pop(2) == [1.0, 2.0]
    assert fifo.get_size() == 1
