from abc import abstractmethod, ABC

from Payload import Payload
from Sink.SinkBehaviour import SinkBehaviour


class Sink(ABC):
    def __init__(self, sink_behaviour: SinkBehaviour):
        self._sink_behaviour: SinkBehaviour = sink_behaviour

    @abstractmethod
    def push(self, payload: Payload) -> None:
        pass
