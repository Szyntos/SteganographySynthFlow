from typing import List

from Data import Text
from Payload import TextPayload
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class TextSink(Sink[TextPayload]):
    def __init__(self, sink_behaviour: SinkBehaviour):
        super().__init__(sink_behaviour)
        self._texts: List[Text] = []
        self._spare_payloads: List[TextPayload] = []

    def push(self, payload: TextPayload) -> None:
        self.collect(payload)

    def collect(self, payload: TextPayload) -> None:
        self._spare_payloads.append(payload)
        self.assemble()

    def assemble(self):
        pass
