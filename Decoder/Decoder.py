from typing import List

from AudioChunk import AudioChunk
from Deserializer import Deserializer
from Payload import SymbolRow
from SamplesFifo import SamplesFifo
from Settings import Settings
from .DecodingStrategy import DecodingStrategy


class Decoder:
    def __init__(
            self,
            settings: Settings,
            decoding_strategy: DecodingStrategy,
            deserializer: Deserializer,
    ):
        self._settings = settings
        self._decoding_strategy: DecodingStrategy = decoding_strategy
        self._deserializer: Deserializer = deserializer
        self._max_driver_block_size: int = 0
        self._audio_chunk_output_fifo: SamplesFifo = SamplesFifo()
        self.reconfigure()

    def reconfigure(self) -> None:
        self._max_driver_block_size = self._settings.max_driver_block_size
        self._audio_chunk_output_fifo = SamplesFifo(
            self._decoding_strategy.get_internal_clock() + self._max_driver_block_size - 1
        )

    def get_output_fifo_size(self) -> int:
        return self._audio_chunk_output_fifo.get_size()

    def process(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        decoded_symbols: List[SymbolRow] = self._decoding_strategy.decode_samples(input_samples, num_samples)
        for symbol_row in decoded_symbols:
            self._audio_chunk_output_fifo.push(
                symbol_row.resample_to_size(self._decoding_strategy.get_internal_clock()))
        self._deserializer.deserialize_symbols(decoded_symbols)
        return AudioChunk(self._audio_chunk_output_fifo.pop_or_silence(num_samples))
