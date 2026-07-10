from abc import abstractmethod
from typing import List

from Framing import FramingSyncController
from Payload import SymbolRow
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class FramedByteSink(Sink):
    """Shared skeleton for sinks that assemble a sync-delimited payload out of
    decoded rows: watch for a start marker, accumulate decoded bytes row by
    row, and finalize when the end marker fires. BinarySink, TextSink and
    ImageSink all differ only in *how* they accumulate and finalize (growable
    buffer + length prefix vs. fixed canvas + merge policy) — this class owns
    the row-by-row control flow all three previously duplicated.
    """

    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        super().__init__(framing_sync_controller, sink_behaviour)
        self._in_frame = False

    def push(self, symbol_row: SymbolRow) -> None:
        self.collect(symbol_row)

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for symbol_row in symbol_rows:
            self.push(symbol_row)

    def collect(self, symbol_row: SymbolRow) -> None:
        self._spare_symbols.append(symbol_row)
        self.assemble()

    def assemble(self) -> None:
        while self._spare_symbols:
            self._process_row(self._spare_symbols.pop(0))

    def on_signal_drop(self) -> None:
        self._reset_state()

    def _process_row(self, symbol_row: SymbolRow) -> None:
        offsets = symbol_row.get_offsets()
        bits = FramingSyncController.quantize_row_to_bits(offsets)
        start_fire, end_fire_enter, is_end_match_now = self._framing_sync_controller.push(bits)

        if start_fire:
            self._in_frame = True
            self._start_frame()

        if not self._in_frame:
            return

        if end_fire_enter:
            self._finalize_frame()
            self._reset_state()
            return

        if is_end_match_now:
            return

        self._accumulate(offsets)

    def _reset_state(self) -> None:
        self._in_frame = False
        self._framing_sync_controller.reset()
        self._start_frame()

    @abstractmethod
    def _start_frame(self) -> None:
        """Called when a start marker fires, and after every reset: clear
        whatever per-frame accumulation state this sink keeps."""

    @abstractmethod
    def _accumulate(self, offsets: List[float]) -> None:
        """Called once per in-frame data row with its raw offsets."""

    @abstractmethod
    def _finalize_frame(self) -> None:
        """Called once when the end marker fires: publish the result."""
