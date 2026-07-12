from typing import Callable, Dict, Optional

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Encoder import Encoder, EncodingStrategy, FourSplitEncodingStrategy, TwoSplitEncodingStrategy
from Payload import AudioPayload, BinaryPayload, ImagePayload, TextPayload
from Payload.pixel_codec import make_pixel_codec
from Serializer import AudioSerializer, BinarySerializer, ImageSerializer, TextSerializer
from SerializerMode import SerializerMode
from Settings import Settings

# The 4-split strategy's decode window (chunk_size/4) is half as long as the
# 2-split strategy's (chunk_size/2), so it needs double the chunk_size to
# keep the same DFT bin spacing between harmonics and avoid leaking Hann
# window energy across neighboring harmonics. See EncodingStrategy classes.
_STRATEGY_CLASSES = {
    "two": TwoSplitEncodingStrategy,
    "four": FourSplitEncodingStrategy,
}

_PAYLOAD_KINDS = ("audio", "image", "binary", "text")


def _make_harmonic_generator(settings: Settings) -> AdditiveWaveGenerator:
    gen = AdditiveWaveGenerator(settings)
    gen.set_omegas([float(i + 1) for i in range(settings.total_harmonics)])
    gen.set_phases([0.0] * settings.total_harmonics)
    gen.set_amps([settings.base_amplitude / (i + 1) for i in range(settings.total_harmonics)])
    return gen


class EncoderDSP:
    """Assembles the full encode pipeline (strategy, payload, codec) behind a
    single real-time-adjustable API, so callers never touch Encoder/
    EncodingStrategy/Serializer/Payload objects directly.

    One EncodingStrategy instance is kept per payload kind for the object's
    lifetime, rather than rebuilt on every kind/codec switch: a strategy's
    clock position and its wave generator's harmonic phases track where the
    encoded signal is in its cycle, and a listening decoder (which may be a
    separate process/device) relies on that position staying continuous.
    Rebuilding on every switch would snap it back to zero and desync any
    decoder that isn't retuned immediately after.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings: Settings = settings if settings is not None else Settings()

        self._payload_kind: str = "audio"
        self._strategy_kind: str = "two"
        self._base_chunk_size: int = self.settings.chunk_size
        self._codec_mode: SerializerMode = SerializerMode.DIGITAL
        self._f0: float = 0.0

        self._audio_payload = AudioPayload()
        self._audio_payload.load_from_file(self.settings.modulator_wav_path)

        self._image_path: str = self.settings.image_path
        self._image_codec = make_pixel_codec(self._codec_mode, self.settings)
        self._image_payload = ImagePayload(self.settings, self._image_codec)
        self._image_payload.load_from_file(self._image_path)

        self._binary_path: Optional[str] = None
        self._binary_codec = make_pixel_codec(self._codec_mode, self.settings)
        self._binary_payload = BinaryPayload(self.settings, self._binary_codec)

        self._text_path: Optional[str] = None
        self._text_codec = make_pixel_codec(self._codec_mode, self.settings)
        self._text_payload = TextPayload(self.settings, self._text_codec)

        self._wave_generator = _make_harmonic_generator(self.settings)

        self._strategies: Dict[str, EncodingStrategy] = {
            kind: self._make_strategy_for(kind) for kind in _PAYLOAD_KINDS
        }
        self._encoding_strategy: EncodingStrategy = self._strategies[self._payload_kind]
        self._encoder = Encoder(self._encoding_strategy)

    # ── strategy assembly ────────────────────────────────────────────────────
    def _encoding_cls(self):
        return _STRATEGY_CLASSES[self._strategy_kind]

    def _payload_for(self, kind: str):
        return {
            "audio": self._audio_payload,
            "image": self._image_payload,
            "binary": self._binary_payload,
            "text": self._text_payload,
        }[kind]

    def _make_strategy_for(self, kind: str) -> EncodingStrategy:
        if kind == "audio":
            serializer = AudioSerializer(self.settings, SerializerMode.DIGITAL)
        elif kind == "image":
            serializer = ImageSerializer(self.settings, self._codec_mode)
        elif kind == "binary":
            serializer = BinarySerializer(self.settings, self._codec_mode)
        elif kind == "text":
            serializer = TextSerializer(self.settings, self._codec_mode)
        else:
            raise ValueError(f"Unknown payload kind: {kind}")
        strategy = self._encoding_cls()(self.settings, self._wave_generator, serializer)
        strategy.load_payload(self._payload_for(kind))
        return strategy

    def _rebuild_strategy_for(self, kind: str) -> None:
        """Structural reset (harmonics/clock geometry changed): rebuild just
        that kind's strategy in place, preserving the others' clocks."""
        self._strategies[kind] = self._make_strategy_for(kind)

    def _rebuild_all_strategies(self) -> None:
        for kind in _PAYLOAD_KINDS:
            self._rebuild_strategy_for(kind)
        self._encoding_strategy = self._strategies[self._payload_kind]
        self._encoding_strategy.set_f0(self._f0)
        self._encoder.set_encoding_strategy(self._encoding_strategy)

    # ── real-time controls ───────────────────────────────────────────────────
    def get_strategy_kind(self) -> str:
        return self._strategy_kind

    def set_strategy_kind(self, kind: str) -> None:
        if kind not in _STRATEGY_CLASSES:
            raise ValueError(f"Unknown strategy kind: {kind}")
        self._strategy_kind = kind
        self.settings.set_chunk_size(self._base_chunk_size * self.settings.strategy_chunk_size_multiplier[kind])
        self._wave_generator = _make_harmonic_generator(self.settings)
        self._rebuild_all_strategies()

    def get_payload_kind(self) -> str:
        return self._payload_kind

    def set_payload_kind(self, kind: str) -> None:
        if kind not in _PAYLOAD_KINDS:
            raise ValueError(f"Unknown payload kind: {kind}")
        self._payload_kind = kind
        self._encoding_strategy = self._strategies[kind]
        self._encoding_strategy.set_f0(self._f0)
        self._encoder.set_encoding_strategy(self._encoding_strategy)

    def get_codec_mode(self) -> SerializerMode:
        return self._codec_mode

    def set_codec_mode(self, mode: SerializerMode) -> None:
        self._codec_mode = mode

        self._image_codec = make_pixel_codec(mode, self.settings)
        image_payload = ImagePayload(self.settings, self._image_codec)
        image_payload.load_from_file(self._image_path)
        self._image_payload = image_payload
        self._strategies["image"].load_payload(image_payload)

        self._binary_codec = make_pixel_codec(mode, self.settings)
        binary_payload = BinaryPayload(self.settings, self._binary_codec)
        if self._binary_path is not None:
            binary_payload.load_from_file(self._binary_path)
        self._binary_payload = binary_payload
        self._strategies["binary"].load_payload(binary_payload)

        self._text_codec = make_pixel_codec(mode, self.settings)
        text_payload = TextPayload(self.settings, self._text_codec)
        if self._text_path is not None:
            text_payload.load_from_file(self._text_path)
        self._text_payload = text_payload
        self._strategies["text"].load_payload(text_payload)

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        self.settings.set_bits_per_symbol(int(bits_per_symbol))

        self._image_codec = make_pixel_codec(self._codec_mode, self.settings)
        image_payload = ImagePayload(self.settings, self._image_codec)
        image_payload.load_from_file(self._image_path)
        self._image_payload = image_payload

        self._binary_codec = make_pixel_codec(self._codec_mode, self.settings)
        binary_payload = BinaryPayload(self.settings, self._binary_codec)
        if self._binary_path is not None:
            binary_payload.load_from_file(self._binary_path)
        self._binary_payload = binary_payload

        self._text_codec = make_pixel_codec(self._codec_mode, self.settings)
        text_payload = TextPayload(self.settings, self._text_codec)
        if self._text_path is not None:
            text_payload.load_from_file(self._text_path)
        self._text_payload = text_payload

        self._wave_generator = _make_harmonic_generator(self.settings)
        self._rebuild_all_strategies()

    def load_payload_file(self, file_path: str) -> None:
        if self._payload_kind == "image":
            self._image_path = file_path
            payload = ImagePayload(self.settings, self._image_codec)
            payload.load_from_file(file_path)
            self._image_payload = payload
        elif self._payload_kind == "binary":
            self._binary_path = file_path
            payload = BinaryPayload(self.settings, self._binary_codec)
            payload.load_from_file(file_path)
            self._binary_payload = payload
        elif self._payload_kind == "text":
            self._text_path = file_path
            payload = TextPayload(self.settings, self._text_codec)
            payload.load_from_file(file_path)
            self._text_payload = payload
        else:
            payload = AudioPayload()
            payload.load_from_file(file_path)
            self._audio_payload = payload
        # Re-loading into the existing strategy keeps its _clock_position
        # and the wave generator's phases running, so a listening decoder
        # stays in sync without needing a retune.
        self._encoding_strategy.load_payload(payload)

    def get_payload_path(self, kind: Optional[str] = None) -> Optional[str]:
        kind = kind or self._payload_kind
        return {
            "image": self._image_path,
            "binary": self._binary_path,
            "text": self._text_path,
            "audio": self.settings.modulator_wav_path,
        }[kind]

    def get_position_fraction(self) -> float:
        return self._encoding_strategy.get_position_fraction()

    def set_position_fraction(self, fraction: float) -> None:
        self._encoding_strategy.set_position_fraction(fraction)

    def set_f0(self, f0: float) -> None:
        self._f0 = float(f0)
        self._encoder.set_f0(self._f0)

    def get_f0(self) -> float:
        return self._f0

    # ── processing ────────────────────────────────────────────────────────────
    def process(self, num_samples: int) -> AudioChunk:
        return self._encoder.process(num_samples)
