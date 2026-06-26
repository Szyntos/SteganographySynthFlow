from AudioChunk import AudioChunk
from .EncodingStrategy import EncodingStrategy


class Encoder:
    def __init__(self, encoding_strategy: EncodingStrategy):
        self._encoding_strategy = encoding_strategy

    def set_encoding_strategy(self, encoding_strategy: EncodingStrategy) -> None:
        self._encoding_strategy = encoding_strategy

    def set_f0(self, f0: float) -> None:
        self._encoding_strategy.set_f0(f0)

    def process(self, num_samples: int) -> AudioChunk:
        return self._encoding_strategy.generate_samples(num_samples)
