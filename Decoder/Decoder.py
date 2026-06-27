from typing import List

from AudioChunk import AudioChunk
from SamplesFifo import SamplesFifo
from Deserializer import Deserializer
from Payload import SymbolRow
from Sink import Sink
from .DecodingStrategy import DecodingStrategy


class Decoder:
    def __init__(
            self,
            decoding_strategy: DecodingStrategy,
            deserializer: Deserializer,
            sink: Sink,
            max_driver_block_size: int,
    ):
        self._decoding_strategy: DecodingStrategy = decoding_strategy
        self._deserializer: Deserializer = deserializer
        self._sink: Sink = sink
        self._max_driver_block_size: int = max_driver_block_size
        self._audio_chunk_output_fifo: SamplesFifo = SamplesFifo(self._decoding_strategy.get_internal_clock() + self._max_driver_block_size - 1)

    def process(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        decoded_symbols: List[SymbolRow] = self._decoding_strategy.decode_samples(input_samples, num_samples)
        for symbol_row in decoded_symbols:
            # Each symbol_row is extracted from _internal_clock samples, so it has to be translated to _internal_clock samples
            self._audio_chunk_output_fifo.push(symbol_row.resample_to_size(self._decoding_strategy.get_internal_clock()))
        self._deserializer.deserialize_symbols(decoded_symbols)
        return AudioChunk(self._audio_chunk_output_fifo.pop_or_silence(num_samples))