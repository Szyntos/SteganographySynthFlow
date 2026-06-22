from Payload import AudioPayload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from .Deserializer import Deserializer


class AudioDeserializer(Deserializer[AudioPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_float: int = 1):
        super().__init__(serializer_mode, bits_per_float)

    def deserialize_payload(self, serialized_payload: SerializedPayload) -> AudioPayload:
        pass