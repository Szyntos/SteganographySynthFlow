from typing import Tuple, List
from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from AudioChunkSink import AudioChunkSink
from Payload import SerializedPayload, SymbolRow
from .DecodingStrategy import DecodingStrategy


class TwoSplitDecodingStrategy(DecodingStrategy):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, audio_chunk_sink: AudioChunkSink):
        super().__init__(additive_wave_generator, audio_chunk_sink)
        self._internal_clock: int = 480

    def decode_samples(self, input_samples: AudioChunk, num_samples: int) -> SymbolRow | None:
        self._audio_chunk_sink.push(input_samples)
        to_decode: List[float] = self._audio_chunk_sink.get_n_samples_or_0(self._internal_clock)
        if to_decode:
            # now we have enough samples to decode the data from the audio_chunk
            pass
            return SymbolRow()

        return None
