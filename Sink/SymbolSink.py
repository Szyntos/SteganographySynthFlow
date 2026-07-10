from typing import List, Protocol, runtime_checkable

from Payload import SymbolRow


@runtime_checkable
class SymbolSink(Protocol):
    """Anything that can receive decoded symbol rows: a concrete Sink, or a
    SinkTee fanning out to several of them."""

    def push(self, symbol_row: SymbolRow) -> None: ...

    def push_many(self, symbol_rows: List[SymbolRow]) -> None: ...
