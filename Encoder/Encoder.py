from AudioChunk import AudioChunk
from Codec.EncoderCodec import EncoderCodec
from .EncodingStrategy import EncodingStrategy
from Payload.Payload import Payload
from Payload.SerializedPayload import SerializedPayload


class Encoder:
    def __init__(
        self,
        codec: EncoderCodec,
        encoding_strategy: EncodingStrategy,
        payload: Payload | None = None,
    ):
        self._codec = codec
        self._encoding_strategy = encoding_strategy
        self._f0 = 440.0
        self._payload: Payload | None = None
        self._serialized_payload = SerializedPayload([])

        self._encoding_strategy.set_f0(self._f0)

        if payload is not None:
            self.set_payload(payload)

    def set_codec(self, codec: EncoderCodec) -> None:
        self._codec = codec
        if self._payload is not None:
            self._serialized_payload = self._codec.serializer().serialize_payload(self._payload)
            self._encoding_strategy.set_serialized_payload(self._serialized_payload)

    def set_payload(self, payload: Payload) -> None:
        self._payload = payload
        self._serialized_payload = self._codec.serializer().serialize_payload(payload)
        self._encoding_strategy.set_serialized_payload(self._serialized_payload)

    def set_encoding_strategy(self, encoding_strategy: EncodingStrategy) -> None:
        self._encoding_strategy = encoding_strategy
        self._encoding_strategy.set_f0(self._f0)
        self._encoding_strategy.set_serialized_payload(self._serialized_payload)

    def set_f0(self, f0: float) -> None:
        self._f0 = f0
        self._encoding_strategy.set_f0(self._f0)

    def process(self, num_samples: int) -> AudioChunk:
        return self._encoding_strategy.generate_samples(num_samples)
