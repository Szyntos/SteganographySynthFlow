from typing import Callable, List, Optional

from Payload import SymbolRow
from Payload.pixel_codec import PixelCodec


class RawTextSink:
    """Framing-free live preview: every decoded row is appended straight onto a
    rolling byte buffer and re-decoded as UTF-8, showing only the last
    `max_chars` characters (sync markers and all). Runs in parallel with the
    framed TextSink to show what the decoder sees before frame
    synchronisation, one character at a time as bytes arrive."""

    # UTF-8 code points are at most 4 bytes; over-allocate the raw buffer so a
    # trailing partial multi-byte sequence never truncates a real character.
    _BYTES_PER_CHAR = 4

    def __init__(self,
                 codec: PixelCodec,
                 max_chars: int = 200,
                 on_text: Optional[Callable[[str], None]] = None):
        self._codec = codec
        self._max_chars = max_chars
        self._max_bytes = max_chars * self._BYTES_PER_CHAR
        self._on_text = on_text
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

    def get_latest_text(self) -> str:
        return bytes(self._buffer).decode("utf-8", errors="ignore")[-self._max_chars:]

    def set_on_text(self, on_text: Optional[Callable[[str], None]]) -> None:
        self._on_text = on_text

    def _publish(self) -> None:
        if self._on_text is not None:
            self._on_text(self.get_latest_text())
