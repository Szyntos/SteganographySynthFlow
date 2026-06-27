import math
from typing import List

from Framing import FramingSyncController
from Payload import SymbolRow
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class AudioSink(Sink):
    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        super().__init__(framing_sync_controller, sink_behaviour)

    def push(self, symbol_row: SymbolRow) -> None:
        pass

    def push_many(self, symbol_rows: List[SymbolRow]) -> None:
        for symbol_row in symbol_rows:
            self.push(symbol_row)