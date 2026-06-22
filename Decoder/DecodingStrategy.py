from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from abc import ABC, abstractmethod


class DecodingStrategy(ABC):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._f0: float = 0.0

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    @abstractmethod
    def generate_samples(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        pass