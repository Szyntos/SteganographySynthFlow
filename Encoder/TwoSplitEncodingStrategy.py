from typing import List

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Payload import SymbolRow
from Serializer import Serializer
from .EncodingStrategy import EncodingStrategy


class TwoSplitEncodingStrategy(EncodingStrategy):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer, num_rows: int):
        super().__init__(additive_wave_generator, serializer, num_rows)
        self._internal_clock = 480
        self._phases: int = 2
        self._phase_duration: int = self._internal_clock // self._phases
        self._current_row: SymbolRow | None = None

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
            # encode num_rows onto the AudioChunk
            pass

        return AudioChunk(result)
