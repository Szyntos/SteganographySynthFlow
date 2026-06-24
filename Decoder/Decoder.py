from AudioChunk import AudioChunk
from Codec.DecoderCodec import DecoderCodec
from .DecodingStrategy import DecodingStrategy


class Decoder:
    def __init__(
            self,
            codec: DecoderCodec,
            decoding_strategy: DecodingStrategy,
    ):
        self._codec: DecoderCodec = codec
        self._decoding_strategy: DecodingStrategy = decoding_strategy

    def set_codec(self, codec: DecoderCodec) -> None:
        self._codec = codec

    def process(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        serialized = self._decoding_strategy.decode_samples(input_samples, num_samples)
        payload = self._codec.deserializer().deserialize_payload(serialized)
        return self._codec.sink().push(payload)
