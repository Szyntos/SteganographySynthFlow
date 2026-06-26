from typing import List

from AudioChunk import AudioChunk
from Framing import FramingSyncController
from Payload import TextPayload, SymbolRow
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class TextSink(Sink[TextPayload]):
    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        super().__init__(framing_sync_controller, sink_behaviour)
        self._texts: List[TextPayload] = []
        self._spare_symbols: List[SymbolRow] = []

    def push(self, payload: SymbolRow) -> AudioChunk:
        self.collect(payload)
        return AudioChunk.silence(123)

    def collect(self, payload: SymbolRow) -> None:
        self._spare_symbols.append(payload)
        self.assemble()

    def assemble(self):
        pass
