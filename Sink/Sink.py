from abc import abstractmethod, ABC
from typing import List

from AudioChunk import AudioChunk
from Framing import FramingSyncController
from Payload import SymbolRow, Payload
from Sink import SinkBehaviour


class Sink(ABC):
    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        self._framing_sync_controller: FramingSyncController = framing_sync_controller
        self._sink_behaviour: SinkBehaviour = sink_behaviour
        self._payloads: List[Payload] = []
        self._spare_symbols: List[SymbolRow] = []

    @abstractmethod
    def push(self, payload: SymbolRow) -> AudioChunk:
        pass
