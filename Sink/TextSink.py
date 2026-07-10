import struct
from typing import Callable, List, Optional

from Framing import FramingSyncController
from Payload import SymbolRow
from Payload.pixel_codec import PixelCodec
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class TextSink(Sink):
    """Codec-agnostic variable-length text accumulator serving both Digital and Analogue.

    Framed "clean" output: publishes once per completed, sync-delimited frame,
    i.e. the previously fully-decoded payload (mirrors ImageSink's canvas)."""

    def __init__(self,
                 framing_sync_controller: FramingSyncController,
                 sink_behaviour: SinkBehaviour,
                 codec: PixelCodec,
                 on_text: Optional[Callable[[str], None]] = None):
        super().__init__(framing_sync_controller, sink_behaviour)
        self._codec = codec
        self._on_text = on_text
        self._latest_text: Optional[str] = None
        self._in_frame = False
        self._buffer = bytearray()

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
        self._latest_text = None
        self._reset_state()

    def get_text(self) -> Optional[str]:
        return self._latest_text

    def set_on_text(self, on_text: Optional[Callable[[str], None]]) -> None:
        self._on_text = on_text

    def _process_row(self, symbol_row: SymbolRow) -> None:
        offsets = symbol_row.get_offsets()
        bits = FramingSyncController.quantize_row_to_bits(offsets)
        start_fire, end_fire_enter, is_end_match_now = self._framing_sync_controller.push(bits)

        if start_fire:
            self._in_frame = True
            self._buffer = bytearray()

        if not self._in_frame:
            return

        if end_fire_enter:
            self._finalize_frame()
            self._reset_state()
            return

        if is_end_match_now:
            return

        self._buffer.extend(self._codec.decode_chunk(offsets))

    def _finalize_frame(self) -> None:
        if len(self._buffer) < 4:
            return
        length = struct.unpack(">I", bytes(self._buffer[:4]))[0]
        available = min(length, len(self._buffer) - 4)
        raw = bytes(self._buffer[4:4 + available])
        self._latest_text = raw.decode("utf-8", errors="replace")
        if self._on_text is not None:
            self._on_text(self._latest_text)

    def _reset_state(self) -> None:
        self._in_frame = False
        self._framing_sync_controller.reset()
        self._buffer = bytearray()
