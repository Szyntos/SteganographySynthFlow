from Payload import BinaryPayload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from . import Serializer


class BinarySerializer(Serializer[BinaryPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_float: int = 1):
        super().__init__(serializer_mode, bits_per_float)

    def serialize_payload(self, serialized_payload: SerializedPayload) -> BinaryPayload:
        pass