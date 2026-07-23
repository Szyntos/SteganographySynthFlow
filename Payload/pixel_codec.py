from typing import List, Protocol, runtime_checkable

from SerializerMode import SerializerMode
from Settings import Settings


def bits_from_bytes_msb_first(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits


def bytes_from_bits_msb_first(bits: List[int], num_bytes: int) -> bytes:
    out = bytearray(num_bytes)
    for i in range(num_bytes):
        value = 0
        for b in range(8):
            index = i * 8 + b
            bit = bits[index] if index < len(bits) else 0
            value = (value << 1) | (bit & 1)
        out[i] = value
    return bytes(out)


@runtime_checkable
class PixelCodec(Protocol):
    """Pure byte-row <-> float-row conversion. Knows nothing about images,
    framing or sync."""

    chunk_size: int

    def encode_chunk(self, pixel_bytes: bytes) -> List[float]:
        ...

    def decode_chunk(self, row: List[float]) -> bytes:
        ...


class DigitalPixelCodec:
    """Packs bits_per_symbol bits per harmonic, MSB-first, quantized to
    2**bits_per_symbol evenly-spaced levels in [-1, 1]."""

    def __init__(self, bits_per_symbol: int, data_harmonics: int,
                 bits_per_chunk: int = None, no_data_epsilon: float = 0.03):
        if not 1 <= bits_per_symbol <= 32:
            raise ValueError("DigitalPixelCodec: bits_per_symbol must be in [1, 32]")
        self._bits_per_symbol = bits_per_symbol
        self._data_harmonics = data_harmonics
        self._bits_per_chunk = bits_per_chunk if bits_per_chunk is not None else data_harmonics * bits_per_symbol
        self._no_data_epsilon = no_data_epsilon
        self._max_v = (1 << bits_per_symbol) - 1
        self.chunk_size = self._bits_per_chunk // 8
        if self.chunk_size == 0:
            raise ValueError("DigitalPixelCodec: bits_per_chunk < 8 (no full byte capacity)")

    def _v_to_level(self, v: int) -> float:
        return -1.0 + 2.0 * (v / self._max_v)

    def _level_to_v(self, x: float) -> int:
        if abs(x) <= self._no_data_epsilon:
            return 0
        xc = min(max(x, -1.0), 1.0)
        return int((xc + 1.0) * 0.5 * self._max_v + 0.5)

    def encode_chunk(self, pixel_bytes: bytes) -> List[float]:
        chunk = bytes(pixel_bytes[: self.chunk_size])
        if len(chunk) < self.chunk_size:
            chunk += b"\x00" * (self.chunk_size - len(chunk))

        bits = bits_from_bytes_msb_first(chunk)
        if len(bits) < self._bits_per_chunk:
            bits.extend([0] * (self._bits_per_chunk - len(bits)))

        row: List[float] = []
        for h in range(self._data_harmonics):
            v = 0
            for b in range(self._bits_per_symbol):
                v = (v << 1) | bits[h * self._bits_per_symbol + b]
            row.append(self._v_to_level(v))
        return row

    def decode_chunk(self, row: List[float]) -> bytes:
        bits: List[int] = []
        for h in range(self._data_harmonics):
            v = self._level_to_v(row[h]) if h < len(row) else 0
            for b in range(self._bits_per_symbol):
                shift = self._bits_per_symbol - 1 - b
                bits.append((v >> shift) & 1)
        return bytes_from_bits_msb_first(bits, self.chunk_size)


class AnaloguePixelCodec:
    """One pixel byte per harmonic, mapped directly as v = -1 + 2*(byte/255)."""

    def __init__(self, data_harmonics: int):
        self.chunk_size = data_harmonics

    def encode_chunk(self, pixel_bytes: bytes) -> List[float]:
        row: List[float] = []
        for i in range(self.chunk_size):
            byte = pixel_bytes[i] if i < len(pixel_bytes) else 0
            row.append(-1.0 + 2.0 * (byte / 255.0))
        return row

    def decode_chunk(self, row: List[float]) -> bytes:
        out = bytearray(self.chunk_size)
        for i in range(self.chunk_size):
            v = row[i] if i < len(row) else -1.0
            v = min(max(v, -1.0), 1.0)
            u = (v + 1.0) * 0.5
            out[i] = min(255, max(0, int(u * 255.0 + 0.5)))
        return bytes(out)


class AudioDigitalCodec:
    """Splits one harmonic's fixed bits_per_symbol budget evenly across
    `samples_per_symbol` raw audio samples (each in [-1, 1]), MSB-first:
    `depth = bits_per_symbol // samples_per_symbol` bits per sample. The
    packed value never exceeds the harmonic's original bits_per_symbol
    range, so more samples per symbol trades amplitude resolution for
    sample-rate density within the same physical quantization budget."""

    def __init__(self, bits_per_symbol: int, samples_per_symbol: int):
        if samples_per_symbol < 1 or (samples_per_symbol & (samples_per_symbol - 1)) != 0:
            raise ValueError("AudioDigitalCodec: samples_per_symbol must be a power of 2")
        depth = bits_per_symbol // samples_per_symbol
        if depth < 1:
            raise ValueError(
                "AudioDigitalCodec: samples_per_symbol too large for bits_per_symbol")
        self._samples_per_symbol = samples_per_symbol
        self._depth = depth
        self._max_v = (1 << (depth * samples_per_symbol)) - 1
        self._max_sub_v = (1 << depth) - 1
        self.chunk_size = samples_per_symbol

    @property
    def samples_per_symbol(self) -> int:
        return self._samples_per_symbol

    def _sample_to_level(self, x: float) -> int:
        xc = min(max(x, -1.0), 1.0)
        return int((xc + 1.0) * 0.5 * self._max_sub_v + 0.5)

    def _level_to_sample(self, v: int) -> float:
        return -1.0 + 2.0 * (v / self._max_sub_v)

    def encode_chunk(self, samples: List[float]) -> List[float]:
        chunk = list(samples[: self.chunk_size])
        if len(chunk) < self.chunk_size:
            chunk += [0.0] * (self.chunk_size - len(chunk))
        v = 0
        for x in chunk:
            v = (v << self._depth) | self._sample_to_level(x)
        return [-1.0 + 2.0 * (v / self._max_v)]

    def decode_chunk(self, row: List[float]) -> List[float]:
        x = row[0] if row else -1.0
        xc = min(max(x, -1.0), 1.0)
        v = int((xc + 1.0) * 0.5 * self._max_v + 0.5)
        out: List[float] = []
        for i in range(self._samples_per_symbol):
            shift = self._depth * (self._samples_per_symbol - 1 - i)
            out.append(self._level_to_sample((v >> shift) & self._max_sub_v))
        return out


def make_pixel_codec(serializer_mode: SerializerMode, settings: Settings) -> PixelCodec:
    if serializer_mode == SerializerMode.DIGITAL:
        return DigitalPixelCodec(
            settings.bits_per_symbol, settings.data_harmonics,
            settings.bits_per_chunk, settings.pixel_codec_no_data_epsilon,
        )
    if serializer_mode == SerializerMode.ANALOGUE:
        return AnaloguePixelCodec(settings.data_harmonics)
    raise ValueError(f"make_pixel_codec: unsupported mode {serializer_mode}")
