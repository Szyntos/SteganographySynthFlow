from typing import List

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Payload import SymbolRow
from Serializer import Serializer
from Settings import Settings
from .EncodingStrategy import EncodingStrategy


class TwoSplitEncodingStrategy(EncodingStrategy):
    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer):
        self._phases: int = 2
        self._phase_duration: int = 0
        self._current_row: SymbolRow | None = None
        super().__init__(settings, additive_wave_generator, serializer)

    def reconfigure(self) -> None:
        super().reconfigure()
        self._phase_duration = self._internal_clock // self._phases
        self._current_row = None

    def generate_samples(self, num_samples: int) -> AudioChunk:
        result: List[float] = []
        for sample in range(num_samples):
            if self._clock_position < self._phase_duration:
                self._current_row = None
                result.append(self._additive_wave_generator.generate_next(self._f0))
            else:
                if self._current_row is None:
                    self._current_row = self._serializer.get_symbol_row(self._num_rows)
                result.append(self._additive_wave_generator.generate_next_with_offsets(self._f0, self._current_row))

            self._clock_position = (self._clock_position + 1) % self._internal_clock

        return AudioChunk(result)
