from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Serializer import Serializer
from .EncodingStrategy import EncodingStrategy


class TwoSplitEncodingStrategy(EncodingStrategy):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer, num_rows: int):
        super().__init__(additive_wave_generator, serializer, num_rows)
        self._internal_clock = 480

    def generate_samples(self, num_samples: int) -> AudioChunk:
        for sample in range(num_samples):
            # encode num_rows onto the AudioChunk
            pass

        return AudioChunk([i**3 for i in range(num_samples)])
