from Payload import TextPayload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from .Deserializer import Deserializer


class TextDeserializer(Deserializer[TextPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        super().__init__(serializer_mode, bits_per_symbol)

    def deserialize_payload(self, serialized_payload: SerializedPayload) -> TextPayload:
        pass