from Payload import ImagePayload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from .Serializer import Serializer


class ImageSerializer(Serializer[ImagePayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        super().__init__(serializer_mode, bits_per_symbol)

    def serialize_payload(self, payload: ImagePayload) -> SerializedPayload:
        pass