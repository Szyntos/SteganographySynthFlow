from Deserializer.ImageDeserializer import ImageDeserializer
from Framing import FramingSyncController
from Payload.ImagePayload import ImagePayload
from Serializer.ImageSerializer import ImageSerializer
from SerializerMode import SerializerMode
from Sink import SinkBehaviour
from Sink.ImageSink import ImageSink
from .DecoderCodec import DecoderCodec
from .EncoderCodec import EncoderCodec


class ImageEncoderCodec(EncoderCodec[ImagePayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._serializer = ImageSerializer(serializer_mode, bits_per_symbol)

    def serializer(self) -> ImageSerializer:
        return self._serializer


class ImageDecoderCodec(DecoderCodec[ImagePayload]):
    def __init__(self,
                 serializer_mode: SerializerMode,
                 bits_per_symbol: int,
                 sink_behaviour: SinkBehaviour,
                 framing_sync_controller: FramingSyncController
                 ):
        self._deserializer = ImageDeserializer(serializer_mode, bits_per_symbol)
        self._sink = ImageSink(framing_sync_controller, sink_behaviour)

    def deserializer(self) -> ImageDeserializer:
        return self._deserializer

    def sink(self) -> ImageSink:
        return self._sink
