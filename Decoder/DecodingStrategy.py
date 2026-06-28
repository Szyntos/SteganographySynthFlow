from abc import ABC, abstractmethod
from typing import List

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Payload import SymbolRow
from SamplesFifo import SamplesFifo
from Settings import Settings


class DecodingStrategy(ABC):
    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator):
        self._settings = settings
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._internal_clock: int = 0
        self._num_rows: int = 0
        self._audio_chunk_input_fifo: SamplesFifo = SamplesFifo()
        self.reconfigure()
        self._f0: float = 440.0

    def reconfigure(self) -> None:
        self._num_rows = self._settings.data_harmonics
        self._internal_clock = self._settings.chunk_size

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_f0(self, f0: float):
        self._f0 = f0

    def get_internal_clock(self) -> int:
        return self._internal_clock

    def decode_samples(self, input_samples: AudioChunk, num_samples: int) -> List[SymbolRow]:
        self._audio_chunk_input_fifo.push(input_samples.get_samples())

        decoded_symbols: List[SymbolRow] = []

        while self._audio_chunk_input_fifo.can_read(self._internal_clock):
            to_decode: List[float] = self._audio_chunk_input_fifo.pop_or_empty(self._internal_clock)
            decoded_symbols.append(self._decode(to_decode))
        return decoded_symbols

    @abstractmethod
    def _decode(self, samples: List[float]) -> SymbolRow:
        pass
