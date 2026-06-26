from AudioChunk import AudioChunk
from Deserializer import Deserializer
from Sink import Sink
from .DecodingStrategy import DecodingStrategy


class Decoder:
    def __init__(
            self,
            decoding_strategy: DecodingStrategy,
            deserializer: Deserializer,
            sink: Sink,
    ):
        self._decoding_strategy: DecodingStrategy = decoding_strategy
        self._deserializer: Deserializer = deserializer
        self._sink: Sink = sink

    def process(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        serialized = self._decoding_strategy.decode_samples(input_samples, num_samples)
        payload = self._deserializer.deserialize_payload(serialized)
        return self._sink.push(payload)
