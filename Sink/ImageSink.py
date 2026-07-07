from typing import Callable, List, Optional, Tuple

from Framing import FramingSyncController
from Payload import SymbolRow
from Payload.pixel_codec import PixelCodec
from Settings import Settings
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour
from .temporal_merge_policy import TemporalMergePolicy

# (pixels, width, height, channels)
ImageFrame = Tuple[bytes, int, int, int]


class ImageSink(Sink):
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

        self._in_frame = False
        self._canvas = bytearray(self._expected_bytes)
        self._arrived = bytearray(self._expected_bytes)
        self._write_offset = 0

    def push(self, symbol_row: SymbolRow) -> None:
        self.collect(symbol_row)

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for symbol_row in symbol_rows:
            self.push(symbol_row)

    def collect(self, symbol_row: SymbolRow) -> None:
        self._spare_symbols.append(symbol_row)
        self.assemble()

    def assemble(self):
        while self._spare_symbols:
            self._process_row(self._spare_symbols.pop(0))

    def on_signal_drop(self) -> None:
        if self._merge_policy is not None and self._in_frame and any(self._arrived):
            self._merge_policy.merge(self._canvas, self._arrived)
            self._publish_persist()
        self._reset_state()

    def get_latest_image(self) -> Optional[ImageFrame]:
        return self._latest_image

    def set_on_image(self, on_image: Optional[Callable[[ImageFrame], None]]) -> None:
        self._on_image = on_image

    def _process_row(self, symbol_row: SymbolRow) -> None:
        offsets = symbol_row.get_offsets()
        bits = FramingSyncController.quantize_row_to_bits(offsets)
        start_fire, end_fire_enter, is_end_match_now = self._framing_sync_controller.push(bits)

        if start_fire:
            self._in_frame = True
            self._clear_canvas()

        if not self._in_frame:
            return

        if end_fire_enter:
            self._finalize_frame()
            self._reset_state()
            return

        if is_end_match_now:
            return

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

    def _clear_canvas(self) -> None:
        self._canvas = bytearray(self._expected_bytes)
        self._arrived = bytearray(self._expected_bytes)
        self._write_offset = 0

    def _reset_state(self) -> None:
        self._in_frame = False
        self._framing_sync_controller.reset()
        self._clear_canvas()
