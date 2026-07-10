from typing import Callable, Optional

from Framing import FramingSyncController
from Payload.pixel_codec import PixelCodec
from .BufferedFramedSink import BufferedFramedSink
from .SinkBehaviour import SinkBehaviour


class TextSink(BufferedFramedSink[str]):
    """Codec-agnostic variable-length text accumulator serving both Digital and Analogue.

    Framed "clean" output: publishes once per completed, sync-delimited frame,
    i.e. the previously fully-decoded payload (mirrors ImageSink's canvas)."""

    def __init__(self,
                 framing_sync_controller: FramingSyncController,
                 sink_behaviour: SinkBehaviour,
                 codec: PixelCodec,
                 on_text: Optional[Callable[[str], None]] = None):
        super().__init__(framing_sync_controller, sink_behaviour, codec, on_result=on_text)

    def _transform(self, raw: bytes) -> str:
        return raw.decode("utf-8", errors="replace")

    def get_text(self) -> Optional[str]:
        return self.get_result()

    def set_on_text(self, on_text: Optional[Callable[[str], None]]) -> None:
        self.set_on_result(on_text)
