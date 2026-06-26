from Deserializer.BinaryDeserializer import BinaryDeserializer
from Framing import FramingSyncController
from Payload.BinaryPayload import BinaryPayload
from Serializer.BinarySerializer import BinarySerializer
from SerializerMode import SerializerMode
from Sink import SinkBehaviour
from Sink.BinarySink import BinarySink
from .DecoderCodec import DecoderCodec
from .EncoderCodec import EncoderCodec


class BinaryEncoderCodec(EncoderCodec[BinaryPayload]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._serializer = BinarySerializer(serializer_mode, bits_per_symbol)

    def serializer(self) -> BinarySerializer:
        return self._serializer


class BinaryDecoderCodec(DecoderCodec[BinaryPayload]):
    def __init__(self,
                 serializer_mode: SerializerMode,
                 bits_per_symbol: int,
                 sink_behaviour: SinkBehaviour,
                 framing_sync_controller: FramingSyncController
                 ):
        self._deserializer = BinaryDeserializer(serializer_mode, bits_per_symbol)
        self._sink = BinarySink(framing_sync_controller, sink_behaviour)

    def deserializer(self) -> BinaryDeserializer:
        return self._deserializer

    def sink(self) -> BinarySink:
        return self._sink
