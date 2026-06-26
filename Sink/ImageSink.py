from typing import List

from Framing import FramingSyncController
from Payload import Payload, SymbolRow
from . import SinkBehaviour
from .Sink import Sink


class ImageSink(Sink):
    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        super().__init__(framing_sync_controller, sink_behaviour)

    def push(self, payload: SymbolRow) -> None:
        self.collect(payload)

    def collect(self, payload: SymbolRow) -> None:
        self._spare_symbols.append(payload)
        self.assemble()

    def assemble(self):
        pass
