from abc import ABC, abstractmethod
from itertools import islice, cycle
from typing import Optional

from Payload import SymbolRow
from Payload.Payload import Payload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from Settings import Settings


class Serializer(ABC):
    def __init__(self, settings: Settings, serializer_mode: SerializerMode):
        self._settings = settings
        self._serializer_mode: SerializerMode = serializer_mode
        self._bits_per_symbol: int = 1
        self._payload: Optional[Payload] = None
        self._serialized_payload: SerializedPayload = SerializedPayload([])
        self._symbol_index = 0
        self.reconfigure()

    def reconfigure(self) -> None:
        self._bits_per_symbol = self._settings.bits_per_symbol
        self._symbol_index = 0

    def get_symbol_row(self, num_symbols: int) -> SymbolRow:
        result: SymbolRow = SymbolRow(list(islice(cycle(self._serialized_payload.get_offsets()), self._symbol_index,
                                                  self._symbol_index + num_symbols)))
        self._symbol_index = (self._symbol_index + num_symbols) % self._serialized_payload.get_size()
        return result

    def reset_loop(self) -> None:
        self._symbol_index = 0

    @abstractmethod
    def load_payload(self, payload: Payload) -> None:
        pass
