from typing import List

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from EncodingStrategy import EncodingStrategy
from Payload import Payload


class Encoder:
    def __init__(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator: AdditiveWaveGenerator = additive_wave_generator
        self._f0: float = 440.0
        self._payload: Payload = Payload()
        self._encoding_strategy: EncodingStrategy = EncodingStrategy(self._additive_wave_generator)
        pass

    def set_payload(self, payload: Payload):
        self._payload = payload
        self._encoding_strategy.set_payload(self._payload)

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._additive_wave_generator = additive_wave_generator

    def set_encoding_strategy(self, encoding_strategy: EncodingStrategy):
        self._encoding_strategy = encoding_strategy
        self._encoding_strategy.set_f0(self._f0)
        self._encoding_strategy.set_payload(self._payload)

    def set_f0(self, f0: float):
        self._f0 = f0
        self._encoding_strategy.set_f0(self._f0)

    def process(self, num_samples: int) -> AudioChunk:
        return self._encoding_strategy.generate_samples(num_samples)