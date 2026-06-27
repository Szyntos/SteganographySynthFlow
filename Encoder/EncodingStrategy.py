from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from abc import ABC, abstractmethod

from Payload.Payload import Payload
from Serializer import Serializer


class EncodingStrategy(ABC):
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer, num_rows: int):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._serializer: Serializer = serializer
        self._num_rows: int = num_rows
        self._f0: float = 0.0
        self._internal_clock = 1
        self._clock_position = 0

    def set_num_rows(self, num_rows: int):
        self._num_rows = num_rows

    def load_payload(self, payload: Payload) -> None:
        self._serializer.load_payload(payload)

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_f0(self, f0: float):
        self._f0 = f0

    @abstractmethod
    def generate_samples(self, num_samples: int) -> AudioChunk:
        pass