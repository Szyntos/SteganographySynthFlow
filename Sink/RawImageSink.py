from typing import Callable, List, Optional

from Payload import SymbolRow
from Payload.pixel_codec import PixelCodec
from Settings import Settings
from .ImageSink import ImageFrame


class RawImageSink:
    """Framing-free live preview: every decoded row is written straight onto a
    rolling canvas (wrapping at the image size), sync markers and all. Runs in
    parallel with the framed ImageSink to show what the decoder sees before
    frame synchronisation."""

    def __init__(self,
                 codec: PixelCodec,
                 settings: Settings,
                 on_image: Optional[Callable[[ImageFrame], None]] = None):
        self._codec = codec
        self._width = settings.image_target_w
        self._height = settings.image_target_h
        self._channels = settings.image_channels
        self._expected_bytes = self._width * self._height * self._channels

        self._on_image = on_image
        self._latest_image: Optional[ImageFrame] = None
        self._canvas = bytearray(self._expected_bytes)
        self._write_offset = 0

    def push(self, symbol_row: SymbolRow) -> None:
        data = self._codec.decode_chunk(symbol_row.get_offsets())
        for byte in data:
            self._canvas[self._write_offset] = byte
            self._write_offset = (self._write_offset + 1) % self._expected_bytes

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for symbol_row in symbol_rows:
            self.push(symbol_row)
        if symbol_rows:
            self._publish()

    def get_latest_image(self) -> Optional[ImageFrame]:
        return self._latest_image

    def set_on_image(self, on_image: Optional[Callable[[ImageFrame], None]]) -> None:
        self._on_image = on_image

    def _publish(self) -> None:
        frame: ImageFrame = (bytes(self._canvas), self._width, self._height, self._channels)
        self._latest_image = frame
        if self._on_image is not None:
            self._on_image(frame)


class SinkTee:
    """Fans push/push_many out to several sinks (duck-typed)."""

    def __init__(self, *sinks):
        self._sinks = sinks

    def push(self, symbol_row: SymbolRow) -> None:
        for sink in self._sinks:
            sink.push(symbol_row)

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for sink in self._sinks:
            sink.push_many(symbol_rows)
