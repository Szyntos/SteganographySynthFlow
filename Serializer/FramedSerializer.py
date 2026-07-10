from Framing.FrameGenerator import FrameGenerator
from Payload import Payload
from Payload.SerializedPayload import SerializedPayload
from SerializerMode import SerializerMode
from Settings import Settings
from .Serializer import Serializer


class FramedSerializer(Serializer):
    """Wraps a payload's data between a start and end sync marker. Shared by
    BinarySerializer, TextSerializer and ImageSerializer, which differ only
    in name — the payload itself already carries any codec/length-prefix
    framing it needs (see Payload._encode_with_codec)."""

    def __init__(self, settings: Settings, serializer_mode: SerializerMode):
        super().__init__(settings, serializer_mode)
        self._frame_generator: FrameGenerator = FrameGenerator(self._settings)

    def load_payload(self, payload: Payload) -> None:
        self._payload = payload
        frame_start = self._frame_generator.get_start()
        frame_end = self._frame_generator.get_end()
        self._serialized_payload = SerializedPayload(frame_start + payload.get_data() + frame_end)
