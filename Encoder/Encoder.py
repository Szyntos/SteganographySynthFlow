from AudioChunk import AudioChunk
from .EncodingStrategy import EncodingStrategy
from Payload.Payload import Payload
from Payload.SerializedPayload import SerializedPayload
from Serializer import Serializer


class Encoder:
    def __init__(
        self,
        serializer: Serializer,
        encoding_strategy: EncodingStrategy,
        payload: Payload | None = None,
    ):
        self._serializer = serializer
        self._encoding_strategy = encoding_strategy
        self._f0 = 440.0
        self._payload = None
        self._serialized_payload = SerializedPayload([])

        self._encoding_strategy.set_f0(self._f0)

        if payload is not None:
            self.set_payload(payload)

    def set_payload(self, payload: Payload):
        self._payload = payload
        self._serialized_payload = self._serializer.serialize_payload(payload)
        self._encoding_strategy.set_serialized_payload(self._serialized_payload)

    def set_serializer(self, serializer: Serializer):
        self._serializer = serializer
        if self._payload is not None:
            self.set_payload(self._payload)

    def set_encoding_strategy(self, encoding_strategy: EncodingStrategy):
        self._encoding_strategy = encoding_strategy
        self._encoding_strategy.set_f0(self._f0)
        self._encoding_strategy.set_serialized_payload(self._serialized_payload)

    def set_f0(self, f0: float):
        self._f0 = f0
        self._encoding_strategy.set_f0(self._f0)

    def process(self, num_samples: int) -> AudioChunk:
        return self._encoding_strategy.generate_samples(num_samples)