from abc import ABC
from typing import List

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Framing.SplitLayout import SplitLayout
from Payload import SymbolRow
from Payload.Payload import Payload
from Serializer import Serializer
from Settings import Settings


class EncodingStrategy(ABC):
    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer):
        self._settings = settings
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._serializer: Serializer = serializer
        self._num_rows: int = 0
        self._internal_clock: int = 0
        self._f0: float = 0.0
        self._clock_position = 0
        self._current_row: SymbolRow | None = None
        self._layout: SplitLayout | None = None
        self.reconfigure()

    def reconfigure(self) -> None:
        self._num_rows = self._settings.data_harmonics
        self._internal_clock = self._settings.chunk_size
        self._clock_position = 0
        self._current_row = None

    def load_payload(self, payload: Payload) -> None:
        self._serializer.load_payload(payload)

    def get_position_fraction(self) -> float:
        return self._serializer.get_position_fraction()

    def set_position_fraction(self, fraction: float) -> None:
        self._serializer.set_position_fraction(fraction)

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_f0(self, f0: float):
        self._f0 = f0

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
            segment_envelope = self._layout.envelope[self._clock_position:self._clock_position + segment_len]
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
