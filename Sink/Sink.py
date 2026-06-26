from abc import abstractmethod, ABC
from typing import Generic, TypeVar

from AudioChunk import AudioChunk
from Framing import FramingSyncController
from Payload import Payload, SymbolRow
from Sink import SinkBehaviour

T = TypeVar('T', bound=Payload)


class Sink(ABC, Generic[T]):
    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        self._framing_sync_controller: FramingSyncController = framing_sync_controller
        self._sink_behaviour: SinkBehaviour = sink_behaviour

    @abstractmethod
    def push(self, payload: SymbolRow) -> AudioChunk:
        pass
