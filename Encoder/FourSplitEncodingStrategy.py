from typing import List

import numpy as np

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Payload import SymbolRow
from Serializer import Serializer
from Settings import Settings
from .EncodingStrategy import EncodingStrategy


class FourSplitEncodingStrategy(EncodingStrategy):
    """Splits each chunk into 4 quarters: pilot, ramp-up, data, ramp-down.

    Quarter 0 carries no phase offset (pilot). Quarter 1 linearly ramps the
    phase offset envelope from 0 to 1 (pilot -> data). Quarter 2 holds the
    full phase offset (data). Quarter 3 linearly ramps the envelope back
    from 1 to 0 (data -> pilot), so the chunk begins and ends on the pilot
    phase, matching the C++ EncodeSplit4 envelope.
    """

    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer):
        self._phases: int = 4
        self._quarter: int = 0
        self._envelope: np.ndarray = np.zeros(0)
        self._current_row: SymbolRow | None = None
        super().__init__(settings, additive_wave_generator, serializer)

    def reconfigure(self) -> None:
        super().reconfigure()
        self._quarter = self._internal_clock // self._phases
        self._envelope = self._build_envelope(self._quarter, self._internal_clock)
        self._current_row = None

    @staticmethod
    def _build_envelope(q: int, chunk_size: int) -> np.ndarray:
        env = np.zeros(chunk_size, dtype=np.float64)
        if q == 0:
            return env

        n = np.arange(q, dtype=np.float64)
        if q > 1:
            env[q:2 * q] = n / q
        else:
            env[q:2 * q] = 1.0

        env[2 * q:3 * q] = 1.0

        if q > 1:
            env[3 * q:4 * q] = 1.0 - n / q
        else:
            env[3 * q:4 * q] = 0.0

        return env

    def _get_phase_offsets(self) -> List[float]:
        if self._current_row is None:
            self._current_row = self._serializer.get_symbol_row(self._num_rows)
        raw: List[float] = self._current_row.get_offsets()
        data_offset: int = self._settings.data_offset
        padded: List[float] = [0.0] * data_offset + raw
        return padded

    def generate_samples(self, num_samples: int) -> AudioChunk:
        result: List[float] = []
        remaining = num_samples

        while remaining > 0:
            if self._clock_position == 0:
                self._current_row = None

            segment_len = min(remaining, self._internal_clock - self._clock_position)
            segment_envelope = self._envelope[self._clock_position:self._clock_position + segment_len]
            block = self._additive_wave_generator.generate_block_with_offsets(
                self._f0,
                segment_len,
                phase_offsets=self._get_phase_offsets(),
                phase_envelope=segment_envelope,
            )

            result.extend(block.tolist())
            self._clock_position = (self._clock_position + segment_len) % self._internal_clock
            remaining -= segment_len

        return AudioChunk(result)
