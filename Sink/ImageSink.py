from typing import Callable, List, Optional, Tuple

from Framing import FramingSyncController
from Payload.pixel_codec import PixelCodec
from Settings import Settings
from .FramedByteSink import FramedByteSink
from .SinkBehaviour import SinkBehaviour
from .temporal_merge_policy import TemporalMergePolicy

# (pixels, width, height, channels)
ImageFrame = Tuple[bytes, int, int, int]


class ImageSink(FramedByteSink):
    """Codec-agnostic canvas accumulator serving both Digital and Analogue."""

    def __init__(self,
                 framing_sync_controller: FramingSyncController,
                 sink_behaviour: SinkBehaviour,
                 codec: PixelCodec,
                 settings: Settings,
                 merge_policy: Optional[TemporalMergePolicy] = None,
                 on_image: Optional[Callable[[ImageFrame], None]] = None):
        super().__init__(framing_sync_controller, sink_behaviour)
        self._codec = codec
        self._width = settings.image_target_w
        self._height = settings.image_target_h
        self._channels = settings.image_channels
        self._expected_bytes = self._width * self._height * self._channels

        self._merge_policy: Optional[TemporalMergePolicy] = None
        if sink_behaviour == SinkBehaviour.CLEAN:
            self._merge_policy = merge_policy if merge_policy is not None else TemporalMergePolicy()

        self._on_image = on_image
        self._latest_image: Optional[ImageFrame] = None

        self._canvas = bytearray(self._expected_bytes)
        self._arrived = bytearray(self._expected_bytes)
        self._write_offset = 0

    def on_signal_drop(self) -> None:
        if self._merge_policy is not None and self._in_frame and any(self._arrived):
            self._merge_policy.merge(self._canvas, self._arrived)
            self._publish_persist()
        self._reset_state()

    def get_latest_image(self) -> Optional[ImageFrame]:
        return self._latest_image

    def set_on_image(self, on_image: Optional[Callable[[ImageFrame], None]]) -> None:
        self._on_image = on_image

    def _start_frame(self) -> None:
        self._canvas = bytearray(self._expected_bytes)
        self._arrived = bytearray(self._expected_bytes)
        self._write_offset = 0

    def _accumulate(self, offsets: List[float]) -> None:
        if self._write_offset >= self._expected_bytes:
            if self._merge_policy is None:
                self._publish_canvas()
            return

        data = self._codec.decode_chunk(offsets)
        remaining = self._expected_bytes - self._write_offset
        take = min(len(data), remaining)
        if take > 0:
            self._canvas[self._write_offset:self._write_offset + take] = data[:take]
            self._arrived[self._write_offset:self._write_offset + take] = b"\x01" * take
            self._write_offset += take

        if self._merge_policy is None:
            self._publish_canvas()

    def _finalize_frame(self) -> None:
        if self._merge_policy is not None:
            self._merge_policy.merge(self._canvas, self._arrived)
            self._publish_persist()
        else:
            self._publish_canvas()

    def _publish_canvas(self) -> None:
        self._publish(bytes(self._canvas))

    def _publish_persist(self) -> None:
        persist = self._merge_policy.persist if self._merge_policy is not None else None
        self._publish(bytes(persist) if persist is not None else bytes(self._canvas))

    def _publish(self, pixels: bytes) -> None:
        frame: ImageFrame = (pixels, self._width, self._height, self._channels)
        self._latest_image = frame
        if self._on_image is not None:
            self._on_image(frame)
