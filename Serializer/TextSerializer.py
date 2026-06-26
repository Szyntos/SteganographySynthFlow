from typing import List

from Framing.FrameGenerator import FrameGenerator
from Payload import TextPayload, SymbolRow
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from .Serializer import Serializer


class TextSerializer(Serializer[TextPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        super().__init__(serializer_mode, bits_per_symbol)
        self._frame_generator: FrameGenerator = FrameGenerator()

    def load_payload(self, payload: TextPayload) -> None:
        self._payload = payload
        frame_start: List[float] = self._frame_generator.get_start()
        frame_end: List[float] = self._frame_generator.get_end()
        self._serialized_payload = SerializedPayload(frame_start + self._payload.get_data() + frame_end)