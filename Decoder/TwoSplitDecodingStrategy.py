from typing import Tuple
from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Payload import SerializedPayload
from .DecodingStrategy import DecodingStrategy


class TwoSplitDecodingStrategy(DecodingStrategy):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator):
        super().__init__(additive_wave_generator)

    def decode_samples(self, input_samples: AudioChunk, num_samples: int) -> SerializedPayload:
        return SerializedPayload([0.2 for _ in range(num_samples)])
