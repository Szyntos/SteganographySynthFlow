from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Payload.SerializedPayload import SerializedPayload
from abc import ABC, abstractmethod

from Serializer import Serializer


class EncodingStrategy(ABC):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._serializer: Serializer = serializer
        self._f0: float = 0.0

    def set_serializer(self, serializer: Serializer):
        self._serializer = serializer

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_f0(self, f0: float):
        self._f0 = f0

    @abstractmethod
    def generate_samples(self, num_samples: int) -> AudioChunk:
        pass # This will encode onto the audio chunk data got from serializer.get_symbol_row()