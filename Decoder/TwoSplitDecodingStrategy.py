import math
from typing import List

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Payload import SymbolRow
from Settings import Settings
from .DecodingStrategy import DecodingStrategy


class TwoSplitDecodingStrategy(DecodingStrategy):
    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator):
        super().__init__(settings, additive_wave_generator)

    def _decode(self, samples: List[float]) -> SymbolRow:
        step: int = self._internal_clock // self._num_rows
        return SymbolRow([samples[i * step] * math.sin(i/100) for i in range(self._num_rows)])
