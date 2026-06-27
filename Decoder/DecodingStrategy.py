from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from abc import ABC, abstractmethod

from AudioChunkSink import AudioChunkSink
from Payload import SerializedPayload, SymbolRow


class DecodingStrategy(ABC):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, audio_chunk_sink: AudioChunkSink):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._audio_chunk_sink: AudioChunkSink = audio_chunk_sink

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_audio_chunk_sink(self, audio_chunk_sink: AudioChunkSink):
        self._audio_chunk_sink = audio_chunk_sink

    @abstractmethod
    def decode_samples(self, input_samples: AudioChunk, num_samples: int) -> SymbolRow | None:
        pass