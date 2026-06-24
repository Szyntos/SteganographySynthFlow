from Deserializer.TextDeserializer import TextDeserializer
from Payload.TextPayload import TextPayload
from Serializer.TextSerializer import TextSerializer
from SerializerMode import SerializerMode
from Sink.TextSink import TextSink
from .DecoderCodec import DecoderCodec
from .EncoderCodec import EncoderCodec


class TextEncoderCodec(EncoderCodec[TextPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._serializer = TextSerializer(serializer_mode, bits_per_symbol)

    def serializer(self) -> TextSerializer:
        return self._serializer


class TextDecoderCodec(DecoderCodec[TextPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int, sink: TextSink):
        self._deserializer = TextDeserializer(serializer_mode, bits_per_symbol)
        self._sink = sink

    def deserializer(self) -> TextDeserializer:
        return self._deserializer

    def sink(self) -> TextSink:
        return self._sink
