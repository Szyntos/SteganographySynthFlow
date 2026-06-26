from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Codec.EncoderCodec import EncoderCodec
from .EncodingStrategy import EncodingStrategy
from Payload.Payload import Payload
from Payload.SerializedPayload import SerializedPayload


class Encoder:
    def __init__(
        self,
        codec: EncoderCodec,
        encoding_strategy: EncodingStrategy,
        payload: Payload | None = None,
    ):
        self._codec = codec
        self._encoding_strategy = encoding_strategy
        self._f0 = 440.0
        self._payload: Payload | None = None

        self._sync_params()

        if payload is not None:
            self.set_payload(payload)



    def _sync_params(self) -> None:
        if self._payload is not None:
            self._codec.serializer().load_payload(self._payload)
        self._encoding_strategy.set_f0(self._f0)
        self._encoding_strategy.set_serializer(self._codec.serializer())


    def set_codec(self, codec: EncoderCodec) -> None:
        self._codec = codec
        self._sync_params()

    def set_payload(self, payload: Payload) -> None:
        self._payload = payload
        self._sync_params()

    def set_encoding_strategy(self, encoding_strategy: EncodingStrategy) -> None:
        self._encoding_strategy = encoding_strategy
        self._sync_params()

    def set_additive_wave_generator(self, additive_wave_generator: AdditiveWaveGenerator):
        self._encoding_strategy.set_additive_wave_generator(additive_wave_generator)

    def set_f0(self, f0: float) -> None:
        self._f0 = f0
        self._sync_params()

    def process(self, num_samples: int) -> AudioChunk:
        return self._encoding_strategy.generate_samples(num_samples)
