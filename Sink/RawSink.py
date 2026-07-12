from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Optional, TypeVar

from Payload import SymbolRow
from Payload.pixel_codec import PixelCodec

T = TypeVar("T")


class RawSink(ABC, Generic[T]):
    """Framing-free live preview: every decoded row is appended straight onto
    a rolling buffer, sync markers and all, so the caller can see what the
    decoder sees before frame synchronisation. Not a Sink subclass — it skips
    framing entirely, so it only needs to duck-type push/push_many.

    Subclasses provide `_cap` (how to keep the buffer bounded) and `_render`
    (how to turn the raw buffer into the published result type)."""

    def __init__(self, codec: PixelCodec, on_result: Optional[Callable[[T], None]] = None):
        self._codec = codec
        self._on_result = on_result
        self._buffer = bytearray()
        self._latest_result: Optional[T] = None

    def push(self, symbol_row: SymbolRow) -> None:
        self._buffer.extend(self._codec.decode_chunk(symbol_row.get_offsets()))
        self._cap()

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for symbol_row in symbol_rows:
            self.push(symbol_row)
        if symbol_rows:
            self._publish()

    def get_latest(self) -> Optional[T]:
        return self._latest_result

    def set_on_result(self, on_result: Optional[Callable[[T], None]]) -> None:
        self._on_result = on_result

    def _publish(self) -> None:
        self._latest_result = self._render()
        if self._on_result is not None:
            self._on_result(self._latest_result)

    def on_signal_drop(self) -> None:
        """No-op: RawSink is a framing-free preview with no frame state to
        reset on a signal drop. Present so callers that treat sinks
        uniformly (e.g. via SymbolSink) don't need to special-case it."""

    @abstractmethod
    def _cap(self) -> None:
        """Keep self._buffer bounded to whatever window this sink previews."""

    @abstractmethod
    def _render(self) -> T:
        """Turn the current buffer into the published result type."""
