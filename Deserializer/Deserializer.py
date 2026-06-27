from abc import ABC, abstractmethod

from Payload import Payload, SymbolRow
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode

class Deserializer(ABC):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._serializer_mode: SerializerMode = serializer_mode
        self._bits_per_symbol: int = bits_per_symbol

    def set_bits_per_symbol(self, bits_per_symbol: int):
        self._bits_per_symbol = bits_per_symbol

    def set_serializer_mode(self, serializer_mode: SerializerMode):
        self._serializer_mode = serializer_mode

    @abstractmethod
    def deserialize_symbols(self, serialized_payload: SerializedPayload) -> SymbolRow:
        pass
