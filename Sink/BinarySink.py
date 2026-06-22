from typing import List

from Data import Binary
from Payload import Payload
from . import SinkBehaviour
from .Sink import Sink


class BinarySink(Sink):
    def __init__(self, sink_behaviour: SinkBehaviour):
        super().__init__(sink_behaviour)
        self._binaries: List[Binary] = []
        self._spare_payloads: List[Payload] = []

    def push(self, payload: Payload) -> None:
        self.collect(payload)

    def collect(self, payload: Payload) -> None:
        self._spare_payloads.append(payload)
        self.assemble()

    def assemble(self):
        pass
