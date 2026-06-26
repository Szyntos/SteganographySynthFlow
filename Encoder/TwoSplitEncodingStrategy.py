from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Serializer import Serializer
from .EncodingStrategy import EncodingStrategy


class TwoSplitEncodingStrategy(EncodingStrategy):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer):
        super().__init__(additive_wave_generator, serializer)

    def generate_samples(self, num_samples: int) -> AudioChunk:
        return AudioChunk([i**3 for i in range(num_samples)])
