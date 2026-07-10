from typing import Callable, Optional

from Framing import FramingSyncController
from Payload.pixel_codec import PixelCodec
from .BufferedFramedSink import BufferedFramedSink
from .SinkBehaviour import SinkBehaviour


class BinarySink(BufferedFramedSink[bytes]):
    """Codec-agnostic variable-length byte accumulator serving both Digital and Analogue.

    Framed "clean" output: publishes once per completed, sync-delimited frame,
    i.e. the previously fully-decoded payload (mirrors ImageSink's canvas)."""

    def __init__(self,
                 framing_sync_controller: FramingSyncController,
                 sink_behaviour: SinkBehaviour,
                 codec: PixelCodec,
                 on_data: Optional[Callable[[bytes], None]] = None):
        super().__init__(framing_sync_controller, sink_behaviour, codec, on_result=on_data)

    def _transform(self, raw: bytes) -> bytes:
        return raw

    def get_bytes(self) -> Optional[bytes]:
        return self.get_result()

    def set_on_data(self, on_data: Optional[Callable[[bytes], None]]) -> None:
        self.set_on_result(on_data)
