from typing import List

from Framing import FramingSyncController
from Payload import SymbolRow
from . import SinkBehaviour
from .Sink import Sink


class ImageSink(Sink):
    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        super().__init__(framing_sync_controller, sink_behaviour)

    def push(self, symbol_row: SymbolRow) -> None:
        self.collect(symbol_row)

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for symbol_row in symbol_rows:
            self.push(symbol_row)

    def collect(self, symbol_row: SymbolRow) -> None:
        self._spare_symbols.append(symbol_row)
        self.assemble()

    def assemble(self):
        pass
