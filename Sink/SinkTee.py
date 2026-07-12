from typing import List

from Payload import SymbolRow


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
