from typing import Callable, Optional

from Payload.pixel_codec import PixelCodec
from Settings import Settings
from .ImageSink import ImageFrame
from .RawSink import RawSink


class RawImageSink(RawSink[ImageFrame]):
    """Framing-free live preview: every decoded row is written straight onto a
    rolling canvas (wrapping at the image size), sync markers and all. Runs in
    parallel with the framed ImageSink to show what the decoder sees before
    frame synchronisation."""

    def __init__(self,
                 codec: PixelCodec,
                 settings: Settings,
                 on_image: Optional[Callable[[ImageFrame], None]] = None):
        super().__init__(codec, on_result=on_image)
        self._width = settings.image_target_w
        self._height = settings.image_target_h
        self._channels = settings.image_channels
        self._expected_bytes = self._width * self._height * self._channels
        self._buffer = bytearray(self._expected_bytes)
        self._write_offset = 0

    def get_latest_image(self) -> Optional[ImageFrame]:
        return self.get_latest()

    def set_on_image(self, on_image: Optional[Callable[[ImageFrame], None]]) -> None:
        self.set_on_result(on_image)

    def _cap(self) -> None:
        # The canvas is a fixed-size ring buffer: fold whatever push() just
        # appended past the end back onto the front, wrapping at
        # _write_offset exactly as the original in-place ring write did.
        excess = len(self._buffer) - self._expected_bytes
        if excess <= 0:
            return
        overflow = bytes(self._buffer[self._expected_bytes:])
        del self._buffer[self._expected_bytes:]

        n = len(overflow)
        first = min(n, self._expected_bytes - self._write_offset)
        self._buffer[self._write_offset:self._write_offset + first] = overflow[:first]
        remaining = overflow[first:]
        if remaining:
            self._buffer[0:len(remaining)] = remaining
        self._write_offset = (self._write_offset + n) % self._expected_bytes

    def _render(self) -> ImageFrame:
        return bytes(self._buffer), self._width, self._height, self._channels
