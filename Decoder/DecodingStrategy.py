from typing import List

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from abc import ABC, abstractmethod

from Payload import SymbolRow
from SamplesFifo import SamplesFifo


class DecodingStrategy(ABC):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, num_rows: int):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._internal_clock: int = 1
        self._num_rows: int = num_rows
        self._audio_chunk_input_fifo: SamplesFifo = SamplesFifo()

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def get_internal_clock(self) -> int:
        return self._internal_clock

    def decode_samples(self, input_samples: AudioChunk, num_samples: int) -> List[SymbolRow]:
        self._audio_chunk_input_fifo.push(input_samples.get_samples())

        decoded_symbols: List[SymbolRow] = []

        while self._audio_chunk_input_fifo.can_read(self._internal_clock):
            to_decode: List[float] = self._audio_chunk_input_fifo.pop_or_empty(self._internal_clock)
            # now we have enough samples to decode the data from the audio_chunk
            decoded_symbols.append(self._decode(to_decode))
        return decoded_symbols

    @abstractmethod
    def _decode(self, samples: List[float]) -> SymbolRow:
        pass