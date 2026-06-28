from typing import List

from Payload import SymbolRow
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import Sink
from .Deserializer import Deserializer


class AudioDeserializer(Deserializer):
    def __init__(self, settings: Settings, sink: Sink, serializer_mode: SerializerMode):
        super().__init__(settings, sink, serializer_mode)

    def deserialize_symbols(self, symbols: List[SymbolRow]) -> None:
        self._sink.push_many(symbols)
