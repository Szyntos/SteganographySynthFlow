from typing import List

from Data import Text
from Payload import Payload
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class TextSink(Sink):
    def __init__(self, sink_behaviour: SinkBehaviour):
        super().__init__(sink_behaviour)
        self._texts: List[Text] = []
        self._spare_payloads: List[Payload] = []

    def push(self, payload: Payload) -> None:
        self.collect(payload)

    def collect(self, payload: Payload) -> None:
        self._spare_payloads.append(payload)
        self.assemble()

    def assemble(self):
        pass
