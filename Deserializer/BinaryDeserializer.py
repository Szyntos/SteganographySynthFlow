from Payload import SymbolRow
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from .Deserializer import Deserializer


class BinaryDeserializer(Deserializer):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        super().__init__(serializer_mode, bits_per_symbol)

    def deserialize_symbols(self, serialized_payload: SerializedPayload) -> SymbolRow:
        pass