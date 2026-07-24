"""Exercises the Serializer base-class logic (symbol cycling, position
fraction, loop reset) via the concrete BinarySerializer/BinaryPayload."""

import pytest

from synthflow.Payload import BinaryPayload
from synthflow.Payload.pixel_codec import make_pixel_codec
from synthflow.Serializer import BinarySerializer
from synthflow.core.SerializerMode import SerializerMode
from synthflow.core.Settings import Settings


@pytest.fixture
def test_binary_file(tmp_path):
    path = tmp_path / "test_binary.bin"
    path.write_bytes(bytes(range(64)))
    return str(path)


def build_serializer(test_binary_file):
    settings = Settings()
    codec = make_pixel_codec(SerializerMode.DIGITAL, settings)
    payload = BinaryPayload(settings, codec)
    payload.load_from_file(test_binary_file)
    serializer = BinarySerializer(settings, SerializerMode.DIGITAL)
    serializer.load_payload(payload)
    return settings, serializer


def test_get_symbol_row_empty_payload_returns_zeros():
    settings = Settings()
    serializer = BinarySerializer(settings, SerializerMode.DIGITAL)
    row = serializer.get_symbol_row(5)
    assert row.get_offsets() == [0.0] * 5


def test_get_symbol_row_wraps_cyclically(test_binary_file):
    settings, serializer = build_serializer(test_binary_file)
    size = serializer._serialized_payload.get_size()

    first_pass = serializer.get_symbol_row(size).get_offsets()
    second_pass = serializer.get_symbol_row(size).get_offsets()

    assert first_pass == second_pass


def test_position_fraction_round_trip(test_binary_file):
    settings, serializer = build_serializer(test_binary_file)
    size = serializer._serialized_payload.get_size()

    serializer.set_position_fraction(0.5)
    assert serializer._symbol_index == int(0.5 * size) % size

    frac = serializer.get_position_fraction()
    assert frac == pytest.approx(serializer._symbol_index / size)


def test_position_fraction_clamps_to_unit_interval(test_binary_file):
    settings, serializer = build_serializer(test_binary_file)
    size = serializer._serialized_payload.get_size()

    serializer.set_position_fraction(-1.0)
    assert serializer._symbol_index == 0

    serializer.set_position_fraction(2.0)
    assert serializer._symbol_index == int(1.0 * size) % size


def test_set_position_fraction_noop_on_empty_payload():
    settings = Settings()
    serializer = BinarySerializer(settings, SerializerMode.DIGITAL)
    serializer.set_position_fraction(0.5)
    assert serializer._symbol_index == 0
    assert serializer.get_position_fraction() == 0.0


def test_reset_loop_zeroes_symbol_index(test_binary_file):
    settings, serializer = build_serializer(test_binary_file)
    serializer.get_symbol_row(10)
    assert serializer._symbol_index != 0
    serializer.reset_loop()
    assert serializer._symbol_index == 0
