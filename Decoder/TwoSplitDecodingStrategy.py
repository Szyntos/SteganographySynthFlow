from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from DecodingStrategy import DecodingStrategy


class TwoSplitDecodingStrategy(DecodingStrategy):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator):
        super().__init__(additive_wave_generator)

    def generate_samples(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        return AudioChunk()
