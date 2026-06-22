from Payload import ImagePayload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from . import Serializer


class ImageSerializer(Serializer[ImagePayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_float: int = 1):
        super().__init__(serializer_mode, bits_per_float)

    def serialize_payload(self, serialized_payload: SerializedPayload) -> ImagePayload:
        pass