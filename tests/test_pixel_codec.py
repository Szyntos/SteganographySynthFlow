import random

import pytest

from Payload.pixel_codec import (
    AnaloguePixelCodec,
    DigitalPixelCodec,
    bits_from_bytes_msb_first,
    bytes_from_bits_msb_first,
    make_pixel_codec,
)
from SerializerMode import SerializerMode
from Settings import Settings


class TestBitHelpers:
    def test_round_trip(self):
        data = bytes(range(256))
        bits = bits_from_bytes_msb_first(data)
        assert bytes_from_bits_msb_first(bits, len(data)) == data

    def test_msb_first_order(self):
        assert bits_from_bytes_msb_first(b"\x80") == [1, 0, 0, 0, 0, 0, 0, 0]
        assert bits_from_bytes_msb_first(b"\x01") == [0, 0, 0, 0, 0, 0, 0, 1]


class TestDigitalPixelCodec:
    def make(self):
        return DigitalPixelCodec(bits_per_symbol=2, data_harmonics=49)

    def test_chunk_size(self):
        assert self.make().chunk_size == (49 * 2) // 8  # 12 bytes

    def test_row_length_and_levels(self):
        codec = self.make()
        row = codec.encode_chunk(bytes(range(codec.chunk_size)))
        assert len(row) == 49
        expected_levels = {-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0}
        assert all(any(abs(v - lv) < 1e-9 for lv in expected_levels) for v in row)

    def test_round_trip_exact(self):
        codec = self.make()
        rng = random.Random(1234)
        for _ in range(50):
            payload = bytes(rng.randrange(256) for _ in range(codec.chunk_size))
            assert codec.decode_chunk(codec.encode_chunk(payload)) == payload

    def test_round_trip_extremes(self):
        codec = self.make()
        for payload in (bytes(codec.chunk_size), b"\xff" * codec.chunk_size):
            assert codec.decode_chunk(codec.encode_chunk(payload)) == payload

    def test_short_chunk_zero_padded(self):
        codec = self.make()
        short = b"\xde\xad\xbe\xef"
        decoded = codec.decode_chunk(codec.encode_chunk(short))
        assert decoded == short + bytes(codec.chunk_size - len(short))

    def test_decode_robust_to_noise(self):
        codec = self.make()
        payload = bytes(range(codec.chunk_size))
        row = codec.encode_chunk(payload)
        rng = random.Random(99)
        noisy = [v + rng.uniform(-0.1, 0.1) for v in row]
        assert codec.decode_chunk(noisy) == payload

    def test_bits_per_symbol_one(self):
        codec = DigitalPixelCodec(bits_per_symbol=1, data_harmonics=49)
        assert codec.chunk_size == 6
        payload = b"\xa5\x5a\x00\xff\x12\x34"
        assert codec.decode_chunk(codec.encode_chunk(payload)) == payload


class TestAnaloguePixelCodec:
    def test_chunk_size(self):
        assert AnaloguePixelCodec(data_harmonics=49).chunk_size == 49

    def test_round_trip_exact_all_values(self):
        codec = AnaloguePixelCodec(data_harmonics=49)
        values = list(range(256))
        for offset in range(0, 256, codec.chunk_size):
            payload = bytes(values[offset:offset + codec.chunk_size])
            decoded = codec.decode_chunk(codec.encode_chunk(payload))
            assert decoded[:len(payload)] == payload

    def test_short_chunk_zero_padded(self):
        codec = AnaloguePixelCodec(data_harmonics=49)
        decoded = codec.decode_chunk(codec.encode_chunk(b"\x7f"))
        assert decoded == b"\x7f" + bytes(codec.chunk_size - 1)

    def test_values_clamped(self):
        codec = AnaloguePixelCodec(data_harmonics=3)
        assert codec.decode_chunk([-2.0, 2.0, 0.0]) == bytes([0, 255, 128])


class TestFactory:
    def test_mode_selects_codec(self):
        settings = Settings()
        assert isinstance(make_pixel_codec(SerializerMode.DIGITAL, settings), DigitalPixelCodec)
        assert isinstance(make_pixel_codec(SerializerMode.ANALOGUE, settings), AnaloguePixelCodec)
