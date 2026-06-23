from AudioChunk import AudioChunk
from Payload import AudioPayload
from Sink import Sink
from .DecodingStrategy import DecodingStrategy
from Deserializer.Deserializer import Deserializer


class Decoder:
    def __init__(
            self,
            deserializer: Deserializer,
            decoding_strategy: DecodingStrategy,
            sink: Sink,
    ):
        self._deserializer: Deserializer = deserializer
        self._decoding_strategy: DecodingStrategy = decoding_strategy
        self._sink: Sink = sink

    def process(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        serialized = self._decoding_strategy.decode_samples(input_samples, num_samples)
        payload = self._deserializer.deserialize_payload(serialized)
        self._sink.push(payload)
        return payload.to_audio_chunk()