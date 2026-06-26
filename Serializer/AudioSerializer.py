from Payload import AudioPayload, SymbolRow, SerializedPayload
from SerializerMode import SerializerMode
from .Serializer import Serializer


class AudioSerializer(Serializer[AudioPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        super().__init__(serializer_mode, bits_per_symbol)

    def load_payload(self, payload: AudioPayload) -> None:
        self._payload = payload
        self._serialized_payload = SerializedPayload(payload.get_data())