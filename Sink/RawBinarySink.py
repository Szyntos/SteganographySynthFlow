from typing import Callable, List, Optional

from Payload import SymbolRow
from Payload.pixel_codec import PixelCodec


class RawBinarySink:
    """Framing-free live preview: every decoded row is appended straight onto a
    rolling byte buffer capped to the last `max_bytes`, sync markers and all.
    Runs in parallel with the framed BinarySink to show what the decoder sees
    before frame synchronisation."""

    def __init__(self,
                 codec: PixelCodec,
                 max_bytes: int = 512,
                 on_data: Optional[Callable[[bytes], None]] = None):
        self._codec = codec
        self._max_bytes = max_bytes
        self._on_data = on_data
        self._buffer = bytearray()

    def push(self, symbol_row: SymbolRow) -> None:
        self._buffer.extend(self._codec.decode_chunk(symbol_row.get_offsets()))
        if len(self._buffer) > self._max_bytes:
            del self._buffer[:len(self._buffer) - self._max_bytes]

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for symbol_row in symbol_rows:
            self.push(symbol_row)
        if symbol_rows:
            self._publish()

    def get_latest_bytes(self) -> bytes:
        return bytes(self._buffer)

    def set_on_data(self, on_data: Optional[Callable[[bytes], None]]) -> None:
        self._on_data = on_data

    def _publish(self) -> None:
        if self._on_data is not None:
            self._on_data(bytes(self._buffer))
