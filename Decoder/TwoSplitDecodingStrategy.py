from typing import List

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import SymbolRow
from .DecodingStrategy import DecodingStrategy


class TwoSplitDecodingStrategy(DecodingStrategy):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, num_rows: int):
        super().__init__(additive_wave_generator, num_rows)
        self._internal_clock: int = 480

    def _decode(self, samples: List[float]) -> SymbolRow:
        step: int = self._internal_clock // self._num_rows
        return SymbolRow([samples[i * step] for i in range(self._num_rows)])  # mock
