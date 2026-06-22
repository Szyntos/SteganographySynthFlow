from AudioChunk import AudioChunk
from .DecodingStrategy import DecodingStrategy
from Deserializer.Deserializer import Deserializer


class Decoder:
    def __init__(
            self,
            deserializer: Deserializer,
            decoding_strategy: DecodingStrategy,
    ):
        self._deserializer: Deserializer = deserializer
        self._decoding_strategy: DecodingStrategy = decoding_strategy

    def process(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        return self._decoding_strategy.generate_samples(input_samples, num_samples)