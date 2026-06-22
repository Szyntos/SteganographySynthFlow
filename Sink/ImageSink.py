from typing import List

from Data import Image
from Payload import ImagePayload
from . import SinkBehaviour
from .Sink import Sink


class ImageSink(Sink[ImagePayload]):
    def __init__(self, sink_behaviour: SinkBehaviour):
        super().__init__(sink_behaviour)
        self._images: List[Image] = []
        self._spare_payloads: List[ImagePayload] = []

    def push(self, payload: ImagePayload) -> None:
        self.collect(payload)

    def collect(self, payload: ImagePayload) -> None:
        self._spare_payloads.append(payload)
        self.assemble()

    def assemble(self):
        pass
