from typing import List

from Data import Binary
from Payload import BinaryPayload
from . import SinkBehaviour
from .Sink import Sink


class BinarySink(Sink[BinaryPayload]):
    def __init__(self, sink_behaviour: SinkBehaviour):
        super().__init__(sink_behaviour)
        self._binaries: List[Binary] = []
        self._spare_payloads: List[BinaryPayload] = []

    def push(self, payload: BinaryPayload) -> None:
        self.collect(payload)

    def collect(self, payload: BinaryPayload) -> None:
        self._spare_payloads.append(payload)
        self.assemble()

    def assemble(self):
        pass
