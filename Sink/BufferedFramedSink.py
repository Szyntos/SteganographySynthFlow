import struct
from typing import Callable, Generic, List, Optional, TypeVar

from Framing import FramingSyncController
from Payload.pixel_codec import PixelCodec
from .FramedByteSink import FramedByteSink
from .SinkBehaviour import SinkBehaviour

T = TypeVar("T")


class BufferedFramedSink(FramedByteSink, Generic[T]):
    """Length-prefixed byte buffer assembly, shared by BinarySink and
    TextSink. Rows are decoded into a growable buffer; on frame end, the
    leading 4-byte big-endian length prefix says how many of the buffered
    bytes are real payload (the rest is trailing padding). Subclasses only
    provide `_transform` to turn those raw bytes into their published result
    type (identity for bytes, utf-8 decode for text)."""

    def __init__(self,
                 framing_sync_controller: FramingSyncController,
                 sink_behaviour: SinkBehaviour,
                 codec: PixelCodec,
                 on_result: Optional[Callable[[T], None]] = None):
        super().__init__(framing_sync_controller, sink_behaviour)
        self._codec = codec
        self._on_result = on_result
        self._latest_result: Optional[T] = None
        self._buffer = bytearray()

    def get_result(self) -> Optional[T]:
        return self._latest_result

    def set_on_result(self, on_result: Optional[Callable[[T], None]]) -> None:
        self._on_result = on_result

    def on_signal_drop(self) -> None:
        self._latest_result = None
        super().on_signal_drop()

    def _start_frame(self) -> None:
        self._buffer = bytearray()

    def _accumulate(self, offsets: List[float]) -> None:
        self._buffer.extend(self._codec.decode_chunk(offsets))

    def _finalize_frame(self) -> None:
        if len(self._buffer) < 4:
            return
        length = struct.unpack(">I", bytes(self._buffer[:4]))[0]
        available = min(length, len(self._buffer) - 4)
        raw = bytes(self._buffer[4:4 + available])
        self._latest_result = self._transform(raw)
        if self._on_result is not None:
            self._on_result(self._latest_result)

    def _transform(self, raw: bytes) -> T:
        raise NotImplementedError
