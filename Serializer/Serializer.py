from abc import ABC, abstractmethod
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
        size = self._serialized_payload.get_size()
        if size == 0:
            return SymbolRow([0.] * num_symbols)

        offsets = self._serialized_payload.get_offsets()
        start = self._symbol_index % size
        result = SymbolRow([offsets[(start + i) % size] for i in range(num_symbols)])

        self._symbol_index = (self._symbol_index + num_symbols) % size
        return result

    def reset_loop(self) -> None:
        self._symbol_index = 0

    def get_position_fraction(self) -> float:
        size = self._serialized_payload.get_size()
        if size == 0:
            return 0.0
        return (self._symbol_index % size) / size

    def set_position_fraction(self, fraction: float) -> None:
        size = self._serialized_payload.get_size()
        if size == 0:
            return
        fraction = min(max(fraction, 0.0), 1.0)
        self._symbol_index = int(fraction * size) % size

    @abstractmethod
    def load_payload(self, payload: Payload) -> None:
        """Subclasses intentionally do NOT reset `_symbol_index` here: it is
        reused as-is (mod the new payload's size) so an encoder mid-broadcast
        keeps advancing rather than snapping back to the start. This is only
        meaningful for the *encoder clock* use case (EncoderDSP swapping a
        payload of the same kind while an EncodingStrategy's clock is
        running); it degrades gracefully but arbitrarily when the new
        payload has a very different length than the old one."""
