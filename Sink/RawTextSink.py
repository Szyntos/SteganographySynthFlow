from typing import Callable, Optional

from Payload.pixel_codec import PixelCodec
from .RawSink import RawSink


class RawTextSink(RawSink[str]):
    """Rolling byte buffer re-decoded as UTF-8, showing only the last
    `max_chars` characters (sync markers and all). Runs in parallel with the
    framed TextSink to show what the decoder sees before frame
    synchronisation, one character at a time as bytes arrive."""

    def __init__(self,
                 codec: PixelCodec,
                 max_chars: int = 200,
                 bytes_per_char: int = 4,
                 on_text: Optional[Callable[[str], None]] = None):
        super().__init__(codec, on_result=on_text)
        self._max_chars = max_chars
        # UTF-8 code points are at most bytes_per_char bytes; over-allocate the
        # raw buffer so a trailing partial multi-byte sequence never truncates
        # a real character.
        self._max_bytes = max_chars * bytes_per_char

    def get_latest_text(self) -> str:
        return self._render()

    def set_on_text(self, on_text: Optional[Callable[[str], None]]) -> None:
        self.set_on_result(on_text)

    def _cap(self) -> None:
        if len(self._buffer) > self._max_bytes:
            del self._buffer[:len(self._buffer) - self._max_bytes]

    def _render(self) -> str:
        return bytes(self._buffer).decode("utf-8", errors="ignore")[-self._max_chars:]
