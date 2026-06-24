from typing import Tuple
from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from abc import ABC, abstractmethod

from Payload import SerializedPayload


class DecodingStrategy(ABC):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    @abstractmethod
    def decode_samples(self, input_samples: AudioChunk, num_samples: int) -> SerializedPayload:
        pass