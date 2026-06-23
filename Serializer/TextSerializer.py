from Payload import TextPayload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from .Serializer import Serializer


class TextSerializer(Serializer[TextPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        super().__init__(serializer_mode, bits_per_symbol)

    def serialize_payload(self, payload: TextPayload) -> SerializedPayload:
        pass