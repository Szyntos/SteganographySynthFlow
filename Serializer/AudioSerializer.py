from Payload import Payload, SerializedPayload
from SerializerMode import SerializerMode
from Settings import Settings
from .Serializer import Serializer


class AudioSerializer(Serializer):
    def __init__(self, settings: Settings, serializer_mode: SerializerMode):
        super().__init__(settings, serializer_mode)

    def load_payload(self, payload: Payload) -> None:
        self._payload = payload
        self._serialized_payload = SerializedPayload(payload.get_data())
