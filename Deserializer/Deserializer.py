from abc import ABC, abstractmethod
from typing import List

from Payload import SymbolRow
from SerializerMode import SerializerMode
from Sink import Sink


class Deserializer(ABC):
    def __init__(self, sink: Sink, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._sink: Sink = sink
        self._serializer_mode: SerializerMode = serializer_mode
        self._bits_per_symbol: int = bits_per_symbol

    def set_bits_per_symbol(self, bits_per_symbol: int):
        self._bits_per_symbol = bits_per_symbol

    def set_serializer_mode(self, serializer_mode: SerializerMode):
        self._serializer_mode = serializer_mode

    @abstractmethod
    def deserialize_symbols(self, symbols: List[SymbolRow]) -> None:
        pass
