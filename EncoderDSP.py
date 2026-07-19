from typing import Dict, Optional

import numpy as np

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Encoder import Encoder, EncodingStrategy
from Payload import AudioPayload, BinaryPayload, ImagePayload, TextPayload
from Payload.pixel_codec import make_pixel_codec
from Serializer import AudioSerializer, BinarySerializer, ImageSerializer, TextSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from StrategyKinds import ENCODING_STRATEGY_CLASSES, apply_strategy_kind
from SynthVoice import SynthVoice
from WaveParams import WaveParams

_STRATEGY_CLASSES = ENCODING_STRATEGY_CLASSES

_PAYLOAD_KINDS = ("audio", "image", "binary", "text")
_CODEC_PAYLOAD_KINDS = ("image", "binary", "text")
_CODEC_PAYLOAD_CLASSES = {"image": ImagePayload, "binary": BinaryPayload, "text": TextPayload}


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
        self._codec_mode: SerializerMode = SerializerMode.DIGITAL
        self._f0: float = 0.0
        apply_strategy_kind(self.settings, self._strategy_kind)

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

        self._wave_params: Optional[WaveParams] = None
        self._wave_generator = self._make_wave_generator()
        self._synth_voice = SynthVoice(self.settings)

        self._strategies: Dict[str, EncodingStrategy] = {
            kind: self._make_strategy_for(kind) for kind in _PAYLOAD_KINDS
        }
        self._encoding_strategy: EncodingStrategy = self._strategies[self._payload_kind]
        self._encoder = Encoder(self._encoding_strategy)

    # ── wave shape ───────────────────────────────────────────────────────────
    def _make_wave_generator(self) -> AdditiveWaveGenerator:
        generator = AdditiveWaveGenerator.harmonic(self.settings)
        params = self._wave_params
        if params is not None and len(params.amps) == self.settings.total_harmonics:
            generator.set_omegas(list(params.omegas))
            generator.set_phases(list(params.phases))
            generator.set_amps(list(params.amps))
        return generator

    def get_wave_params(self) -> WaveParams:
        if self._wave_params is not None:
            return self._wave_params
        return WaveParams.harmonic_default(self.settings)

    def set_wave_params(self, params: WaveParams) -> None:
        params.validate()
        if len(params.amps) != self.settings.total_harmonics:
            raise ValueError(
                f"wave params carry {len(params.amps)} harmonics, "
                f"expected {self.settings.total_harmonics}")
        self._wave_params = params
        # Retune the live generator in place: strategies keep their reference,
        # so clock position and running phases stay continuous.
        self._wave_generator.set_omegas(list(params.omegas))
        self._wave_generator.set_phases(list(params.phases))
        self._wave_generator.set_amps(list(params.amps))

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
        strategy.set_synth_voice(self._synth_voice)
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
        apply_strategy_kind(self.settings, kind)
        self._wave_generator = self._make_wave_generator()
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

    def _reload_codec_payload(self, kind: str, mode: SerializerMode):
        """Rebuild the pixel codec + payload for one of the codec-backed
        payload kinds (image/binary/text), reloading from its stored path
        (image always has one; binary/text may not have been given a file
        yet). Updates `self._<kind>_codec`/`self._<kind>_payload` and
        returns the new payload."""
        codec = make_pixel_codec(mode, self.settings)
        payload = _CODEC_PAYLOAD_CLASSES[kind](self.settings, codec)
        path = getattr(self, f"_{kind}_path")
        if path is not None:
            payload.load_from_file(path)
        setattr(self, f"_{kind}_codec", codec)
        setattr(self, f"_{kind}_payload", payload)
        return payload

    def set_codec_mode(self, mode: SerializerMode) -> None:
        self._codec_mode = mode
        for kind in _CODEC_PAYLOAD_KINDS:
            payload = self._reload_codec_payload(kind, mode)
            self._strategies[kind].load_payload(payload)

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        self.settings.set_bits_per_symbol(int(bits_per_symbol))

        for kind in _CODEC_PAYLOAD_KINDS:
            self._reload_codec_payload(kind, self._codec_mode)

        self._wave_generator = self._make_wave_generator()
        self._rebuild_all_strategies()

    def apply_dsp_settings(self, values: dict) -> None:
        """Atomically apply advanced DSP settings (chunk geometry, harmonic
        layout, image shape) and rebuild everything that bakes them in."""
        self.settings.apply_values(values)
        self.rebuild_after_settings_change()

    def rebuild_after_settings_change(self) -> None:
        """Structural rebuild after settings were mutated externally (e.g. the
        linked engine applied them through the other DSP's facade)."""
        # Custom wave params carry one entry per harmonic; a harmonic-count
        # change makes them meaningless, so fall back to the default shape.
        if (self._wave_params is not None
                and len(self._wave_params.amps) != self.settings.total_harmonics):
            self._wave_params = None
        apply_strategy_kind(self.settings, self._strategy_kind)
        for kind in _CODEC_PAYLOAD_KINDS:
            self._reload_codec_payload(kind, self._codec_mode)
        self._wave_generator = self._make_wave_generator()
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

    # ── synth voice (envelopes + filter) ─────────────────────────────────────
    def note_on(self) -> None:
        self._synth_voice.note_on()

    def note_off(self) -> None:
        self._synth_voice.note_off()

    def set_voice_enabled(self, enabled: bool) -> None:
        self._synth_voice.set_enabled(enabled)

    def get_filtered_wave_params(self) -> WaveParams:
        """The carrier as the filter currently shapes it (amp envelope
        excluded so the displayed shape doesn't shrink with volume). The
        true, unmodified wave stays what get_wave_params() returns."""
        params = self.get_wave_params()
        freqs = np.asarray(params.omegas, dtype=np.float64) * self._f0
        gains = self._synth_voice.display_filter_gains(freqs)
        amps = (np.asarray(params.amps, dtype=np.float64) * gains).tolist()
        return WaveParams(amps, list(params.phases), list(params.omegas))

    # ── processing ────────────────────────────────────────────────────────────
    def process(self, num_samples: int) -> AudioChunk:
        return self._encoder.process(num_samples)
