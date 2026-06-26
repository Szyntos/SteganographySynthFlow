from Payload import SymbolRow
from Payload.SerializedPayload import SerializedPayload
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
from itertools import islice, cycle

from Payload.Payload import Payload
from SerializerMode import SerializerMode

T = TypeVar('T', bound=Payload)

class Serializer(ABC, Generic[T]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._serializer_mode: SerializerMode = serializer_mode
        self._bits_per_symbol: int = bits_per_symbol
        self._payload: Optional[T] = None
        self._serialized_payload: SerializedPayload = SerializedPayload([])
        self._symbol_index = 0

    def get_symbol_row(self, num_symbols: int) -> SymbolRow:
        result: SymbolRow = SymbolRow(list(islice(cycle(self._serialized_payload.get_offsets()), self._symbol_index, self._symbol_index + num_symbols)))
        self._symbol_index = (self._symbol_index + num_symbols) % self._serialized_payload.get_size()
        return result

    def reset_loop(self) -> None:
        self._symbol_index = 0

    @abstractmethod
    def load_payload(self, payload: T) -> None:
        pass
