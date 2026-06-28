from typing import List

from Payload import SymbolRow
from SerializerMode import SerializerMode
from Sink import Sink
from .Deserializer import Deserializer


class BinaryDeserializer(Deserializer):
    def __init__(self, sink: Sink, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        super().__init__(sink, serializer_mode, bits_per_symbol)

    def deserialize_symbols(self, symbols: List[SymbolRow]) -> None:
        pass
