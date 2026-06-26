from Deserializer.TextDeserializer import TextDeserializer
from Framing import FramingSyncController
from Payload.TextPayload import TextPayload
from Serializer.TextSerializer import TextSerializer
from SerializerMode import SerializerMode
from Sink import SinkBehaviour
from Sink.TextSink import TextSink
from .DecoderCodec import DecoderCodec
from .EncoderCodec import EncoderCodec


class TextEncoderCodec(EncoderCodec[TextPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._serializer = TextSerializer(serializer_mode, bits_per_symbol)

    def serializer(self) -> TextSerializer:
        return self._serializer


class TextDecoderCodec(DecoderCodec[TextPayload]):
    def __init__(self,
                 serializer_mode: SerializerMode,
                 bits_per_symbol: int,
                 sink_behaviour: SinkBehaviour,
                 framing_sync_controller: FramingSyncController
                 ):
        self._deserializer = TextDeserializer(serializer_mode, bits_per_symbol)
        self._sink = TextSink(framing_sync_controller, sink_behaviour)

    def deserializer(self) -> TextDeserializer:
        return self._deserializer

    def sink(self) -> TextSink:
        return self._sink
