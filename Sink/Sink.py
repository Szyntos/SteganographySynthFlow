from abc import abstractmethod, ABC
from typing import Generic, TypeVar

from AudioChunk import AudioChunk
from Payload import Payload
from SinkBehaviour import SinkBehaviour

T = TypeVar('T', bound=Payload)


class Sink(ABC, Generic[T]):
    def __init__(self, sink_behaviour: SinkBehaviour):
        self._sink_behaviour: SinkBehaviour = sink_behaviour

    @abstractmethod
    def push(self, payload: T) -> AudioChunk:
        pass
