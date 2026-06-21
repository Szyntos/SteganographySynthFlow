from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Payload import Payload


class EncodingStrategy:
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._payload: Payload = Payload()
        self._f0: float = 0.0

    def set_payload(self, payload: Payload):
        self._payload = payload

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_f0(self, f0: float):
        self._f0 = f0

    def generate_samples(self, num_samples: int) -> AudioChunk:
        return AudioChunk()