from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from EncodingStrategy import EncodingStrategy


class TwoSplitEncodingStrategy(EncodingStrategy):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator):
        super().__init__(additive_wave_generator)

    def generate_samples(self, num_samples: int) -> AudioChunk:
        return AudioChunk()
