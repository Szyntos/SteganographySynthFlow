from typing import Callable, Optional

from Payload.pixel_codec import PixelCodec
from .RawSink import RawSink


class RawBinarySink(RawSink[bytes]):
    """Rolling byte buffer capped to the last `max_bytes`, sync markers and
    all. Runs in parallel with the framed BinarySink to show what the
    decoder sees before frame synchronisation."""

    def __init__(self,
                 codec: PixelCodec,
                 max_bytes: int = 512,
                 on_data: Optional[Callable[[bytes], None]] = None):
        super().__init__(codec, on_result=on_data)
        self._max_bytes = max_bytes

    def get_latest_bytes(self) -> bytes:
        return bytes(self._buffer)

    def set_on_data(self, on_data: Optional[Callable[[bytes], None]]) -> None:
        self.set_on_result(on_data)

    def _cap(self) -> None:
        if len(self._buffer) > self._max_bytes:
            del self._buffer[:len(self._buffer) - self._max_bytes]

    def _render(self) -> bytes:
        return bytes(self._buffer)
