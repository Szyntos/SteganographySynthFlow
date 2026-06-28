from abc import ABC, abstractmethod
from typing import List

from Payload import SymbolRow
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import Sink


class Deserializer(ABC):
    def __init__(self, settings: Settings, sink: Sink, serializer_mode: SerializerMode):
        self._settings = settings
        self._sink: Sink = sink
        self._serializer_mode: SerializerMode = serializer_mode
        self._bits_per_symbol: int = 1
        self.reconfigure()

    def reconfigure(self) -> None:
        self._bits_per_symbol = self._settings.bits_per_symbol

    def set_serializer_mode(self, serializer_mode: SerializerMode):
        self._serializer_mode = serializer_mode

    @abstractmethod
    def deserialize_symbols(self, symbols: List[SymbolRow]) -> None:
        pass
