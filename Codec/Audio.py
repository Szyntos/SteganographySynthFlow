from Deserializer.AudioDeserializer import AudioDeserializer
from Payload.AudioPayload import AudioPayload
from Serializer.AudioSerializer import AudioSerializer
from SerializerMode import SerializerMode
from Sink.AudioSink import AudioSink
from .DecoderCodec import DecoderCodec
from .EncoderCodec import EncoderCodec


class AudioEncoderCodec(EncoderCodec[AudioPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._serializer = AudioSerializer(serializer_mode, bits_per_symbol)

    def serializer(self) -> AudioSerializer:
        return self._serializer


class AudioDecoderCodec(DecoderCodec[AudioPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int, sink: AudioSink):
        self._deserializer = AudioDeserializer(serializer_mode, bits_per_symbol)
        self._sink = sink

    def deserializer(self) -> AudioDeserializer:
        return self._deserializer

    def sink(self) -> AudioSink:
        return self._sink
