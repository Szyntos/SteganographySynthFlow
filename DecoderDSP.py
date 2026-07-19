from typing import Callable, List, Optional

from AudioChunk import AudioChunk
from Decoder import Decoder, DecodingStrategy
from DropTolerance import DropAction, DropTolerance, DropToleranceConfig
from EnergyGate import EnergyGate, EnergyGateConfig
from F0Estimator import F0Tracker
from Framing import FramingSyncController
from Payload.pixel_codec import make_pixel_codec
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import (
    AudioSink, BinarySink, ImageSink, RawBinarySink, RawImageSink, RawTextSink,
    SinkBehaviour, SinkTee, TextSink,
)
from StrategyKinds import DECODING_STRATEGY_CLASSES, apply_strategy_kind

_STRATEGY_CLASSES = DECODING_STRATEGY_CLASSES


class DecoderDSP:
    """Assembles the full decode pipeline (strategy, sinks, f0 estimation,
    energy gating, drop tolerance) behind a single real-time-adjustable API,
    so callers never touch Decoder/DecodingStrategy/Sink objects directly.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings: Settings = settings if settings is not None else Settings()

        self._payload_kind: str = "audio"
        self._strategy_kind: str = "two"
        self._codec_mode: SerializerMode = SerializerMode.DIGITAL
        self._sink_behaviour: SinkBehaviour = SinkBehaviour.LIVE
        self._resample_method: str = "poly"
        apply_strategy_kind(self.settings, self._strategy_kind)

        self._on_image: Optional[Callable] = None
        self._image_sink: Optional[ImageSink] = None
        self._on_raw_image: Optional[Callable] = None
        self._raw_image_sink: Optional[RawImageSink] = None
        self._audio_sink: Optional[AudioSink] = None
        self._on_data: Optional[Callable] = None
        self._binary_sink: Optional[BinarySink] = None
        self._on_raw_data: Optional[Callable] = None
        self._raw_binary_sink: Optional[RawBinarySink] = None
        self._on_text: Optional[Callable] = None
        self._text_sink: Optional[TextSink] = None
        self._on_raw_text: Optional[Callable] = None
        self._raw_text_sink: Optional[RawTextSink] = None

        self._image_codec = make_pixel_codec(self._codec_mode, self.settings)
        self._binary_codec = make_pixel_codec(self._codec_mode, self.settings)
        self._text_codec = make_pixel_codec(self._codec_mode, self.settings)

        # FIFO tuning: how many input samples still to drop so the decode
        # window slides to the requested offset within a chunk.
        self._tune_offset: int = 0
        self._pending_skip: int = 0

        self._f0_tracker = F0Tracker(self.settings)

        self._dec_strategy: DecodingStrategy = self._decoding_cls()(self.settings)
        self._dec_strategy.set_f0_resolver(self._resolve_window_f0)
        self._decoder: Optional[Decoder] = None
        self._rebuild_decode_chain()

        self._energy_gate = EnergyGate(EnergyGateConfig(
            ema_alpha=self.settings.energy_gate_ema_alpha, abs_floor=self.settings.energy_gate_abs_floor,
            drop_ratio=self.settings.energy_gate_drop_ratio,
        ))
        self._drop_tolerance = DropTolerance(DropToleranceConfig(
            tolerance_chunks=self.settings.drop_tolerance_chunks,
        ))
        self._is_gated: bool = False
        self.set_f0(self.settings.pitch_default_hz)

    # ── sink/strategy assembly ───────────────────────────────────────────────
    def _decoding_cls(self):
        return _STRATEGY_CLASSES[self._strategy_kind]

    def _build_image_sinks(self):
        sink = ImageSink(
            FramingSyncController.from_settings(self.settings),
            self._sink_behaviour,
            self._image_codec,
            self.settings,
            on_image=self._on_image,
        )
        raw_sink = RawImageSink(self._image_codec, self.settings, on_image=self._on_raw_image)
        return sink, raw_sink

    def _build_binary_sinks(self):
        sink = BinarySink(
            FramingSyncController.from_settings(self.settings),
            self._sink_behaviour,
            self._binary_codec,
            on_data=self._on_data,
        )
        raw_sink = RawBinarySink(
            self._binary_codec, max_bytes=self.settings.raw_binary_sink_max_bytes, on_data=self._on_raw_data,
        )
        return sink, raw_sink

    def _build_text_sinks(self):
        sink = TextSink(
            FramingSyncController.from_settings(self.settings),
            self._sink_behaviour,
            self._text_codec,
            on_text=self._on_text,
        )
        raw_sink = RawTextSink(
            self._text_codec, max_chars=self.settings.raw_text_sink_max_chars,
            bytes_per_char=self.settings.raw_text_sink_bytes_per_char, on_text=self._on_raw_text,
        )
        return sink, raw_sink

    # Per payload-kind: (builder, sink attr name, raw-sink attr name).
    _FRAMED_SINK_BUILDERS = {
        "image": ("_build_image_sinks", "_image_sink", "_raw_image_sink"),
        "binary": ("_build_binary_sinks", "_binary_sink", "_raw_binary_sink"),
        "text": ("_build_text_sinks", "_text_sink", "_raw_text_sink"),
    }

    def _rebuild_decode_chain(self) -> None:
        self._image_sink = None
        self._raw_image_sink = None
        self._audio_sink = None
        self._binary_sink = None
        self._raw_binary_sink = None
        self._text_sink = None
        self._raw_text_sink = None

        spec = self._FRAMED_SINK_BUILDERS.get(self._payload_kind)
        if spec is not None:
            builder_name, sink_attr, raw_sink_attr = spec
            sink, raw_sink = getattr(self, builder_name)()
            setattr(self, sink_attr, sink)
            setattr(self, raw_sink_attr, raw_sink)
            decode_sink = SinkTee(sink, raw_sink)
        else:
            sink = AudioSink(FramingSyncController(), SinkBehaviour.LIVE, self.settings)
            self._audio_sink = sink
            decode_sink = sink

        self._decoder = Decoder(self.settings, self._dec_strategy, decode_sink, self._resample_method)

    # ── real-time controls ───────────────────────────────────────────────────
    def get_strategy_kind(self) -> str:
        return self._strategy_kind

    def set_strategy_kind(self, kind: str) -> None:
        if kind not in _STRATEGY_CLASSES:
            raise ValueError(f"Unknown strategy kind: {kind}")
        self._strategy_kind = kind
        apply_strategy_kind(self.settings, kind)
        self._dec_strategy = self._decoding_cls()(self.settings)
        self._dec_strategy.set_f0_resolver(self._resolve_window_f0)
        self._tune_offset = 0
        self._pending_skip = 0
        self._rebuild_decode_chain()

    def get_payload_kind(self) -> str:
        return self._payload_kind

    def set_payload_kind(self, kind: str) -> None:
        if kind not in ("audio", "image", "binary", "text"):
            raise ValueError(f"Unknown payload kind: {kind}")
        self._payload_kind = kind
        self._dec_strategy.reconfigure()
        self._rebuild_decode_chain()

    def get_codec_mode(self) -> SerializerMode:
        return self._codec_mode

    def set_codec_mode(self, mode: SerializerMode) -> None:
        self._codec_mode = mode
        self._image_codec = make_pixel_codec(mode, self.settings)
        self._binary_codec = make_pixel_codec(mode, self.settings)
        self._text_codec = make_pixel_codec(mode, self.settings)
        if self._payload_kind != "audio":
            self._dec_strategy.reconfigure()
            self._rebuild_decode_chain()

    def set_sink_behaviour(self, behaviour: SinkBehaviour) -> None:
        self._sink_behaviour = behaviour
        if self._payload_kind != "audio":
            self._rebuild_decode_chain()

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        self.settings.set_bits_per_symbol(int(bits_per_symbol))
        self._image_codec = make_pixel_codec(self._codec_mode, self.settings)
        self._binary_codec = make_pixel_codec(self._codec_mode, self.settings)
        self._text_codec = make_pixel_codec(self._codec_mode, self.settings)
        self._dec_strategy.reconfigure()
        self._rebuild_decode_chain()

    def set_resample_method(self, method: str) -> None:
        self._resample_method = method
        if self._decoder is not None:
            self._decoder.set_resample_method(method)

    def set_tune_offset(self, offset: int) -> None:
        """Slide the decode window by dropping input samples.

        Moving the slider by +d drops d samples; moving it by -d is realised
        as dropping (chunk_size - d), so the window always converges to the
        requested offset modulo chunk_size.
        """
        chunk = self.settings.chunk_size
        offset = int(offset) % chunk
        delta = (offset - self._tune_offset) % chunk
        self._tune_offset = offset
        self._pending_skip = (self._pending_skip + delta) % chunk

    def get_tune_offset(self) -> int:
        return self._tune_offset

    # ── callbacks ─────────────────────────────────────────────────────────────
    def set_on_image(self, callback: Optional[Callable]) -> None:
        self._on_image = callback
        if self._image_sink is not None:
            self._image_sink.set_on_image(callback)

    def set_on_raw_image(self, callback: Optional[Callable]) -> None:
        self._on_raw_image = callback
        if self._raw_image_sink is not None:
            self._raw_image_sink.set_on_image(callback)

    def set_on_data(self, callback: Optional[Callable]) -> None:
        self._on_data = callback
        if self._binary_sink is not None:
            self._binary_sink.set_on_data(callback)

    def set_on_raw_data(self, callback: Optional[Callable]) -> None:
        self._on_raw_data = callback
        if self._raw_binary_sink is not None:
            self._raw_binary_sink.set_on_data(callback)

    def set_on_text(self, callback: Optional[Callable]) -> None:
        self._on_text = callback
        if self._text_sink is not None:
            self._text_sink.set_on_text(callback)

    def set_on_raw_text(self, callback: Optional[Callable]) -> None:
        self._on_raw_text = callback
        if self._raw_text_sink is not None:
            self._raw_text_sink.set_on_text(callback)

    # ── outputs ──────────────────────────────────────────────────────────────
    def get_latest_image(self):
        return self._image_sink.get_latest_image() if self._image_sink is not None else None

    def get_latest_bytes(self) -> Optional[bytes]:
        return self._binary_sink.get_bytes() if self._binary_sink is not None else None

    def get_latest_text(self) -> Optional[str]:
        return self._text_sink.get_text() if self._text_sink is not None else None

    def dump_decoded_audio_to_wav(self, file_path: str) -> None:
        if self._audio_sink is None:
            raise RuntimeError("No decoded audio available (switch payload type to Audio).")
        self._audio_sink.dump_to_wav(file_path)

    def dump_decoded_bytes_to_file(self, file_path: str) -> None:
        if self._binary_sink is None or self._binary_sink.get_bytes() is None:
            raise RuntimeError("No decoded binary data available yet.")
        with open(file_path, "wb") as f:
            f.write(self._binary_sink.get_bytes())

    # ── f0 / gating state ────────────────────────────────────────────────────
    def set_f0(self, f0: float) -> None:
        # Windows already (partially) buffered in the strategy FIFO were
        # captured at the old pitch; the new manual f0 must not apply to them.
        chunk = self.settings.chunk_size
        buffered = self._dec_strategy.get_input_fifo_size()
        defer = (buffered + chunk - 1) // chunk
        self._f0_tracker.set_manual_f0(f0, defer_windows=defer)

    def set_f0_estimator_mode(self, mode: str) -> None:
        self._f0_tracker.set_mode(mode)

    def set_pitch_quantize(self, enabled: bool) -> None:
        self._f0_tracker.set_quantize(enabled)

    def get_estimated_f0(self) -> float:
        return self._f0_tracker.f0

    def get_confidence(self) -> float:
        return self._f0_tracker.confidence

    def _resolve_window_f0(self, pilot_samples: List[float], dirty: bool = False) -> float:
        return self._f0_tracker.resolve(pilot_samples, float(self.settings.fs_out), dirty=dirty)

    def get_drop_run(self) -> int:
        return self._drop_tolerance.drop_run

    def is_gated(self) -> bool:
        return self._is_gated

    def _handle_missing(self) -> None:
        action = self._drop_tolerance.push(True)
        if action == DropAction.RESET_NOW:
            self._dec_strategy.reset_decode_state()
            self._f0_tracker.reset()

    def reset(self) -> None:
        self._energy_gate.reset()
        self._drop_tolerance.reset()
        self._is_gated = False

    # ── processing ────────────────────────────────────────────────────────────
    def process_chunk(self, samples: List[float], num_samples: int) -> List[float]:
        """Runs the full decode step for one audio block: FIFO tuning skip,
        energy gating, f0 estimation/quantization, drop tolerance, and the
        actual symbol decode. Returns the decoded audio samples (silence if
        gated or no pitch could be resolved).
        """
        if self._pending_skip > 0:
            drop = min(self._pending_skip, len(samples))
            samples = samples[drop:]
            self._pending_skip -= drop

        rms_now = EnergyGate.rms(samples)
        self._is_gated = self._energy_gate.is_drop(rms_now)
        if self._is_gated:
            self._handle_missing()
            # Mock silence still flows into the decoder so the chunk FIFO
            # keeps advancing: dropping gated blocks would shift window
            # alignment by the gap length and corrupt everything after it.
            self._decoder.process(AudioChunk([0.0] * len(samples)), num_samples, gated=True)
            return [0.0] * num_samples

        # f0 is resolved inside the decoder, once per chunk-aligned window
        # (via the strategy's f0 resolver), so estimation always runs on the
        # pilot segment of the exact window being decoded. The block must be
        # fed to the decoder unconditionally or the chunk FIFO would starve.
        dec_chunk = self._decoder.process(AudioChunk(list(samples)), num_samples)

        if not self._f0_tracker.has_pitch():
            self._handle_missing()
            return [0.0] * num_samples

        self._drop_tolerance.push(False)
        return dec_chunk.get_samples()
