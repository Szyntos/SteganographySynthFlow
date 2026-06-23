from Payload import BinaryPayload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from .Serializer import Serializer


class BinarySerializer(Serializer[BinaryPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        super().__init__(serializer_mode, bits_per_symbol)

    def serialize_payload(self, payload: BinaryPayload) -> SerializedPayload:
        pass