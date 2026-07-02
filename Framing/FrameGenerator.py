from typing import List
import random
import zlib

from Settings import Settings


class FrameGenerator:
    def __init__(self, settings: Settings):
        self._settings: Settings = settings

    def get_start(self) -> List[int]:
        return balanced_sync_bits(self._settings.sync_msg_start, self._settings.bits_per_symbol, self._settings.data_harmonics)

    def get_end(self) -> List[int]:
        return balanced_sync_bits(self._settings.sync_msg_end, self._settings.bits_per_symbol, self._settings.data_harmonics)


def append_value_bits_msb(out: List[int], v: int, bits_per_symbol: int) -> None:
    for i in range(bits_per_symbol):
        shift = bits_per_symbol - 1 - i
        out.append((v >> shift) & 1)


def crc32(value: str) -> int:
    return zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF


def balanced_sync_bits(
    sync_str: str,
    bits_per_symbol: int,
    data_harmonics: int,
) -> List[int]:
    levels = 1 << bits_per_symbol
    max_v = levels - 1

    v_low = max_v // 2
    v_high = min(max_v, v_low + 1)

    idx = list(range(data_harmonics))

    seed = crc32(sync_str)
    rng = random.Random(seed)
    rng.shuffle(idx)

    k = data_harmonics // 2

    values = [v_low] * data_harmonics
    for i in range(k):
        values[idx[i]] = v_high

    bits: List[int] = []

    for h in range(data_harmonics):
        append_value_bits_msb(bits, values[h], bits_per_symbol)

    bits_per_chunk: int = bits_per_symbol * data_harmonics

    if len(bits) < bits_per_chunk:
        bits.extend([0] * (bits_per_chunk - len(bits)))

    if len(bits) > bits_per_chunk:
        bits = bits[:bits_per_chunk]

    return bits