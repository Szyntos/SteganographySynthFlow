from abc import ABC, abstractmethod

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Payload.Payload import Payload
from Serializer import Serializer
from Settings import Settings


class EncodingStrategy(ABC):
    def __init__(self, settings: Settings, additive_wave_generator: AdditiveWaveGenerator, serializer: Serializer):
        self._settings = settings
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._serializer: Serializer = serializer
        self._num_rows: int = 0
        self._internal_clock: int = 0
        self._f0: float = 0.0
        self._clock_position = 0
        self.reconfigure()

    def reconfigure(self) -> None:
        self._num_rows = self._settings.data_harmonics
        self._internal_clock = self._settings.chunk_size
        self._clock_position = 0

    def load_payload(self, payload: Payload) -> None:
        self._serializer.load_payload(payload)

    def get_position_fraction(self) -> float:
        return self._serializer.get_position_fraction()

    def set_position_fraction(self, fraction: float) -> None:
        self._serializer.set_position_fraction(fraction)

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_f0(self, f0: float):
        self._f0 = f0

    @abstractmethod
    def generate_samples(self, num_samples: int) -> AudioChunk:
        pass
