from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from SerializedPayload import SerializedPayload
from abc import ABC, abstractmethod


class EncodingStrategy(ABC):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._serialized_payload: SerializedPayload = SerializedPayload()
        self._f0: float = 0.0

    def set_serialized_payload(self, serialized_payload: SerializedPayload):
        self._serialized_payload = serialized_payload

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_f0(self, f0: float):
        self._f0 = f0

    @abstractmethod
    def generate_samples(self, num_samples: int) -> AudioChunk:
        pass