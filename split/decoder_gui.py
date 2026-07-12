import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from typing import Callable, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # Settings uses paths relative to the project root

import numpy as np
from PIL import Image as PILImage, ImageTk

try:
    import sounddevice as sd
except ImportError:
    sd = None

from audio_callback_diag import log_callback_event
from AudioChunk import AudioChunk
from Decoder import Decoder, DecodingStrategy, FourSplitDecodingStrategy, TwoSplitDecodingStrategy
from DropTolerance import DropAction, DropTolerance, DropToleranceConfig
from EnergyGate import EnergyGate, EnergyGateConfig
from F0Estimator import AutocorrF0Estimator, FFTF0Estimator, quantize_to_chromatic_hz
from Framing import FramingSyncController
from Payload.pixel_codec import make_pixel_codec
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import (
    AudioSink, BinarySink, ImageSink, RawBinarySink, RawImageSink, RawTextSink,
    SinkBehaviour, SinkTee, TextSink,
)


# The 4-split strategy's decode window (chunk_size/4) is half as long as the
# 2-split strategy's (chunk_size/2), so it needs double the chunk_size to
# keep the same DFT bin spacing between harmonics and avoid leaking Hann
# window energy across neighboring harmonics.
_STRATEGY_CLASSES = {
    "two": TwoSplitDecodingStrategy,
    "four": FourSplitDecodingStrategy,
}


def list_devices(kind: str) -> List[Tuple[int, str]]:
    """kind: 'input' or 'output'."""
    if sd is None:
        return []
    key = f"max_{kind}_channels"
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev[key] > 0:
            devices.append((idx, f"[{idx}] {dev['name']}"))
    return devices


class DecoderEngine:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._volume: float = 1.0
        self._stream = None
        self._lock = threading.Lock()
        self._input_device: Optional[int] = None
        self._output_device: Optional[int] = None

        self._payload_kind: str = "audio"
        self._strategy_kind: str = "two"
        self._base_chunk_size: int = settings.chunk_size
        self._codec_mode: SerializerMode = SerializerMode.DIGITAL
        self._sink_behaviour: SinkBehaviour = SinkBehaviour.LIVE
        self._on_image: Optional[Callable] = None
        self._image_sink: Optional[ImageSink] = None
        self._on_raw_image: Optional[Callable] = None
        self._raw_image_sink: Optional[RawImageSink] = None
        self._audio_sink: Optional[AudioSink] = None
        self._image_codec = make_pixel_codec(self._codec_mode, settings)
        self._on_data: Optional[Callable] = None
        self._binary_sink: Optional[BinarySink] = None
        self._on_raw_data: Optional[Callable] = None
        self._raw_binary_sink: Optional[RawBinarySink] = None
        self._binary_codec = make_pixel_codec(self._codec_mode, settings)
        self._on_text: Optional[Callable] = None
        self._text_sink: Optional[TextSink] = None
        self._on_raw_text: Optional[Callable] = None
        self._raw_text_sink: Optional[RawTextSink] = None
        self._text_codec = make_pixel_codec(self._codec_mode, settings)

        # FIFO tuning: how many input samples still to drop so the decode
        # window slides to the requested offset within a chunk.
        self._tune_offset: int = 0
        self._pending_skip: int = 0

        self._dec_strategy = self._decoding_cls()(settings)
        self._decoder: Optional[Decoder] = None
        self._resample_method: str = "poly"
        self._rebuild_decode_chain()

        self._manual_f0: float = settings.pitch_default_hz
        self._f0_mode: str = "manual"  # "manual", "autocorr", "fft"
        self._pitch_quantize: bool = False
        self._autocorr_estimator = AutocorrF0Estimator(
            f_min_hz=settings.autocorr_f0_min_hz, f_max_hz=settings.autocorr_f0_max_hz,
            rms_floor=settings.autocorr_rms_floor, corr_threshold=settings.autocorr_corr_threshold,
        )
        self._fft_estimator = FFTF0Estimator(
            n_fft=settings.fft_f0_n_fft, f_min_hz=settings.fft_f0_min_hz,
            f_max_hz=settings.fft_f0_max_hz, rms_floor=settings.fft_rms_floor,
        )
        self._last_f0_q: float = 0.0
        self._last_confidence: float = 0.0
        self._energy_gate = EnergyGate(EnergyGateConfig(
            ema_alpha=settings.energy_gate_ema_alpha, abs_floor=settings.energy_gate_abs_floor,
            drop_ratio=settings.energy_gate_drop_ratio,
        ))
        self._drop_tolerance = DropTolerance(DropToleranceConfig(
            tolerance_chunks=settings.drop_tolerance_chunks,
        ))
        self._is_gated: bool = False
        self.set_f0(settings.pitch_default_hz)

    def _rebuild_decode_chain(self) -> None:
        self._image_sink = None
        self._raw_image_sink = None
        self._audio_sink = None
        self._binary_sink = None
        self._raw_binary_sink = None
        self._text_sink = None
        self._raw_text_sink = None

        if self._payload_kind == "image":
            sink = ImageSink(
                FramingSyncController.from_settings(self._settings),
                self._sink_behaviour,
                self._image_codec,
                self._settings,
                on_image=self._on_image,
            )
            self._image_sink = sink
            raw_sink = RawImageSink(self._image_codec, self._settings, on_image=self._on_raw_image)
            self._raw_image_sink = raw_sink
            decode_sink = SinkTee(sink, raw_sink)
        elif self._payload_kind == "binary":
            sink = BinarySink(
                FramingSyncController.from_settings(self._settings),
                self._sink_behaviour,
                self._binary_codec,
                on_data=self._on_data,
            )
            self._binary_sink = sink
            raw_sink = RawBinarySink(
                self._binary_codec, max_bytes=self._settings.raw_binary_sink_max_bytes, on_data=self._on_raw_data,
            )
            self._raw_binary_sink = raw_sink
            decode_sink = SinkTee(sink, raw_sink)
        elif self._payload_kind == "text":
            sink = TextSink(
                FramingSyncController.from_settings(self._settings),
                self._sink_behaviour,
                self._text_codec,
                on_text=self._on_text,
            )
            self._text_sink = sink
            raw_sink = RawTextSink(
                self._text_codec, max_chars=self._settings.raw_text_sink_max_chars,
                bytes_per_char=self._settings.raw_text_sink_bytes_per_char, on_text=self._on_raw_text,
            )
            self._raw_text_sink = raw_sink
            decode_sink = SinkTee(sink, raw_sink)
        else:
            sink = AudioSink(FramingSyncController(), SinkBehaviour.LIVE, self._settings)
            self._audio_sink = sink
            decode_sink = sink

        self._decoder = Decoder(self._settings, self._dec_strategy, decode_sink, self._resample_method)

    def _decoding_cls(self):
        return _STRATEGY_CLASSES[self._strategy_kind]

    def set_strategy_kind(self, kind: str) -> None:
        if kind not in _STRATEGY_CLASSES:
            raise ValueError(f"Unknown strategy kind: {kind}")
        with self._lock:
            self._strategy_kind = kind
            self._settings.set_chunk_size(self._base_chunk_size * self._settings.strategy_chunk_size_multiplier[kind])
            self._dec_strategy = self._decoding_cls()(self._settings)
            f0 = self._manual_f0 if self._f0_mode == "manual" else self._last_f0_q
            if f0 > 0.0:
                self._dec_strategy.set_f0(f0)
            self._tune_offset = 0
            self._pending_skip = 0
            self._rebuild_decode_chain()

    # ── public controls ──────────────────────────────────────────────────────
    def set_volume(self, vol: float) -> None:
        self._volume = float(vol)

    def set_input_device(self, device: Optional[int]) -> None:
        self._input_device = device

    def set_output_device(self, device: Optional[int]) -> None:
        self._output_device = device

    def is_running(self) -> bool:
        return self._stream is not None

    def set_payload_kind(self, kind: str) -> None:
        if kind not in ("audio", "image", "binary", "text"):
            raise ValueError(f"Unknown payload kind: {kind}")
        with self._lock:
            self._payload_kind = kind
            self._dec_strategy.reconfigure()
            self._rebuild_decode_chain()

    def set_codec_mode(self, mode: SerializerMode) -> None:
        with self._lock:
            self._codec_mode = mode
            self._image_codec = make_pixel_codec(mode, self._settings)
            self._binary_codec = make_pixel_codec(mode, self._settings)
            self._text_codec = make_pixel_codec(mode, self._settings)
            if self._payload_kind != "audio":
                self._dec_strategy.reconfigure()
                self._rebuild_decode_chain()

    def set_sink_behaviour(self, behaviour: SinkBehaviour) -> None:
        with self._lock:
            self._sink_behaviour = behaviour
            if self._payload_kind != "audio":
                self._rebuild_decode_chain()

    def set_on_image(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._on_image = callback
            if self._image_sink is not None:
                self._image_sink.set_on_image(callback)

    def set_on_raw_image(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._on_raw_image = callback
            if self._raw_image_sink is not None:
                self._raw_image_sink.set_on_image(callback)

    def set_on_data(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._on_data = callback
            if self._binary_sink is not None:
                self._binary_sink.set_on_data(callback)

    def set_on_raw_data(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._on_raw_data = callback
            if self._raw_binary_sink is not None:
                self._raw_binary_sink.set_on_data(callback)

    def set_on_text(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._on_text = callback
            if self._text_sink is not None:
                self._text_sink.set_on_text(callback)

    def set_on_raw_text(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._on_raw_text = callback
            if self._raw_text_sink is not None:
                self._raw_text_sink.set_on_text(callback)

    def get_latest_bytes(self) -> Optional[bytes]:
        with self._lock:
            return self._binary_sink.get_bytes() if self._binary_sink is not None else None

    def get_latest_text(self) -> Optional[str]:
        with self._lock:
            return self._text_sink.get_text() if self._text_sink is not None else None

    def dump_decoded_bytes_to_file(self, file_path: str) -> None:
        with self._lock:
            if self._binary_sink is None or self._binary_sink.get_bytes() is None:
                raise RuntimeError("No decoded binary data available yet.")
            with open(file_path, "wb") as f:
                f.write(self._binary_sink.get_bytes())

    def set_f0(self, f0: float) -> None:
        self._manual_f0 = float(f0)
        if self._f0_mode == "manual":
            self._dec_strategy.set_f0(self._manual_f0)

    def set_f0_estimator_mode(self, mode: str) -> None:
        if mode not in ("manual", "autocorr", "fft"):
            raise ValueError(f"Unknown f0 estimator mode: {mode}")
        with self._lock:
            self._f0_mode = mode
            self._last_f0_q = 0.0
            if mode == "manual":
                self._dec_strategy.set_f0(self._manual_f0)

    def set_pitch_quantize(self, enabled: bool) -> None:
        self._pitch_quantize = bool(enabled)

    def set_resample_method(self, method: str) -> None:
        with self._lock:
            self._resample_method = method
            if self._decoder is not None:
                self._decoder.set_resample_method(method)

    def dump_decoded_audio_to_wav(self, file_path: str) -> None:
        with self._lock:
            if self._audio_sink is None:
                raise RuntimeError("No decoded audio available (switch payload type to Audio).")
            self._audio_sink.dump_to_wav(file_path)

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        with self._lock:
            self._settings.set_bits_per_symbol(int(bits_per_symbol))
            self._image_codec = make_pixel_codec(self._codec_mode, self._settings)
            self._binary_codec = make_pixel_codec(self._codec_mode, self._settings)
            self._text_codec = make_pixel_codec(self._codec_mode, self._settings)
            self._dec_strategy.reconfigure()
            self._rebuild_decode_chain()

    def set_tune_offset(self, offset: int) -> None:
        """Slide the decode window by dropping input samples.

        Moving the slider by +d drops d samples; moving it by -d is realised
        as dropping (chunk_size - d), so the window always converges to the
        requested offset modulo chunk_size.
        """
        chunk = self._settings.chunk_size
        offset = int(offset) % chunk
        with self._lock:
            delta = (offset - self._tune_offset) % chunk
            self._tune_offset = offset
            self._pending_skip = (self._pending_skip + delta) % chunk

    def get_tune_offset(self) -> int:
        return self._tune_offset

    def get_estimated_f0(self) -> float:
        return self._last_f0_q

    def get_confidence(self) -> float:
        return self._last_confidence

    def get_drop_run(self) -> int:
        return self._drop_tolerance.drop_run

    def is_gated(self) -> bool:
        return self._is_gated

    def _handle_missing(self) -> None:
        action = self._drop_tolerance.push(True)
        if action == DropAction.RESET_NOW:
            self._dec_strategy.reconfigure()
            self._autocorr_estimator.reset()
            self._fft_estimator.reset()
            self._last_f0_q = 0.0

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, status) -> None:
        _cb_start = time.perf_counter()
        try:
            with self._lock:
                samples = indata[:, 0].astype(np.float32)

                if self._pending_skip > 0:
                    drop = min(self._pending_skip, len(samples))
                    samples = samples[drop:]
                    self._pending_skip -= drop

                rms_now = EnergyGate.rms(samples)
                self._is_gated = self._energy_gate.is_drop(rms_now)
                if self._is_gated:
                    self._handle_missing()
                    outdata[:, 0] = 0.0
                    return

                if self._f0_mode == "manual":
                    f_hat = self._manual_f0
                    self._last_confidence = 1.0
                else:
                    estimator = (
                        self._autocorr_estimator if self._f0_mode == "autocorr" else self._fft_estimator
                    )
                    f_hat = estimator.estimate(samples, float(self._settings.fs_out))
                    self._last_confidence = estimator.confidence

                if self._pitch_quantize and f_hat > 0.0:
                    f_hat = quantize_to_chromatic_hz(f_hat, self._settings.pitch_quantizer_a4_hz)

                if f_hat > 0.0:
                    self._last_f0_q = f_hat
                elif self._last_f0_q > 0.0:
                    f_hat = self._last_f0_q

                if f_hat <= 0.0:
                    self._handle_missing()
                    outdata[:, 0] = 0.0
                    return

                self._drop_tolerance.push(False)
                self._dec_strategy.set_f0(f_hat)

                dec_chunk = self._decoder.process(AudioChunk(samples.tolist()), frames)
                arr = np.array(dec_chunk.get_samples(), dtype=np.float32) * self._volume
                outdata[:, 0] = arr
        finally:
            duration = time.perf_counter() - _cb_start
            budget = frames / float(self._settings.fs_out)
            log_callback_event("decoder", status, duration, budget)

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed.\nRun: pip install sounddevice")
        self._energy_gate.reset()
        self._drop_tolerance.reset()
        self._is_gated = False
        self._stream = sd.Stream(
            samplerate=self._settings.fs_out,
            blocksize=self._settings.audio_driver_polling_rate,
            channels=1,
            dtype="float32",
            device=(self._input_device, self._output_device),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class DecoderApp(tk.Tk):
    _RESAMPLE_METHOD_TO_KEY = {
        "Polyphase (resample_poly)": "poly",
        "Linear (continuous)": "linear",
        "Zero-order hold (basic)": "hold",
    }

    def __init__(self):
        super().__init__()
        self.title("SSF Decoder")
        self.resizable(False, False)

        settings = Settings()
        settings.validate()
        self._settings = settings
        self._engine = DecoderEngine(settings)
        self._engine.set_on_image(self._on_image_frame)
        self._engine.set_on_raw_image(self._on_raw_image_frame)
        self._engine.set_on_data(self._on_decoded_data)
        self._engine.set_on_raw_data(self._on_raw_decoded_data)
        self._engine.set_on_text(self._on_decoded_text)
        self._engine.set_on_raw_text(self._on_raw_decoded_text)
        self._running = False
        self._pending_image_frame = None
        self._pending_raw_frame = None
        self._pending_decoded_text: Optional[str] = None
        self._pending_raw_decoded_text: Optional[str] = None
        self._input_devices = list_devices("input")
        self._output_devices = list_devices("output")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_image()
        self._poll_decoded_text()
        self._poll_estimated_f0()

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 6}

        self._status_var = tk.StringVar(value="Stopped")
        status_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        status_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 0))
        ttk.Label(status_frame, text="Status:").pack(side="left", padx=6, pady=4)
        ttk.Label(status_frame, textvariable=self._status_var, font=("", 10, "bold")).pack(
            side="left", pady=4
        )
        self._signal_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self._signal_var, foreground="#b00").pack(
            side="left", padx=12, pady=4
        )
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self._toggle_btn = ttk.Button(self, text="▶  Start", command=self._toggle, width=14)
        self._toggle_btn.grid(row=1, column=0, columnspan=2, **pad)

        # ── left column ─────────────────────────────────────────────────────
        left = ttk.Frame(self)
        left.grid(row=2, column=0, sticky="new")
        left.columnconfigure(0, weight=1)

        # ── devices ──────────────────────────────────────────────────────────
        dev_frame = ttk.LabelFrame(left, text="Audio Devices", padding=8)
        dev_frame.grid(row=0, column=0, sticky="ew", **pad)
        dev_frame.columnconfigure(1, weight=1)

        ttk.Label(dev_frame, text="Input:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        in_names = ["(default)"] + [name for _, name in self._input_devices]
        self._in_device_var = tk.StringVar(value=in_names[0])
        in_combo = ttk.Combobox(
            dev_frame, textvariable=self._in_device_var, values=in_names,
            state="readonly", width=38,
        )
        in_combo.grid(row=0, column=1, sticky="ew", pady=(0, 4))
        in_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        ttk.Label(dev_frame, text="Output:").grid(row=1, column=0, sticky="w", padx=(0, 6))
        out_names = ["(default)"] + [name for _, name in self._output_devices]
        self._out_device_var = tk.StringVar(value=out_names[0])
        out_combo = ttk.Combobox(
            dev_frame, textvariable=self._out_device_var, values=out_names,
            state="readonly", width=38,
        )
        out_combo.grid(row=1, column=1, sticky="ew")
        out_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        # ── split strategy ────────────────────────────────────────────────────
        strategy_frame = ttk.LabelFrame(left, text="Split Strategy", padding=8)
        strategy_frame.grid(row=0, column=1, sticky="ew", **pad)

        self._strategy_var = tk.StringVar(value="two")
        ttk.Radiobutton(
            strategy_frame, text="Two-Split", variable=self._strategy_var,
            value="two", command=self._on_strategy_change,
        ).grid(row=0, column=0, padx=8)
        ttk.Radiobutton(
            strategy_frame, text="Four-Split", variable=self._strategy_var,
            value="four", command=self._on_strategy_change,
        ).grid(row=0, column=1, padx=8)

        # ── payload kind ──────────────────────────────────────────────────────
        kind_frame = ttk.LabelFrame(left, text="Payload Type", padding=8)
        kind_frame.grid(row=1, column=0, sticky="ew", **pad)

        self._kind_var = tk.StringVar(value="audio")
        ttk.Radiobutton(
            kind_frame, text="Audio", variable=self._kind_var,
            value="audio", command=self._on_kind_change,
        ).grid(row=0, column=0, padx=8)
        ttk.Radiobutton(
            kind_frame, text="Image", variable=self._kind_var,
            value="image", command=self._on_kind_change,
        ).grid(row=0, column=1, padx=8)
        ttk.Radiobutton(
            kind_frame, text="Binary", variable=self._kind_var,
            value="binary", command=self._on_kind_change,
        ).grid(row=0, column=2, padx=8)
        ttk.Radiobutton(
            kind_frame, text="Text", variable=self._kind_var,
            value="text", command=self._on_kind_change,
        ).grid(row=0, column=3, padx=8)

        # ── fifo tuning ──────────────────────────────────────────────────────
        tune_frame = ttk.LabelFrame(left, text="Window Tuning (samples)", padding=8)
        tune_frame.grid(row=2, column=0, sticky="ew", **pad)

        self._tune_label = ttk.Label(tune_frame, text="0", width=6, anchor="e")
        self._tune_label.grid(row=0, column=1, padx=(6, 0))

        self._tune_slider = ttk.Scale(
            tune_frame, from_=0, to=self._settings.chunk_size - 1,
            orient="horizontal", length=280, command=self._on_tune_change,
        )
        self._tune_slider.set(0)
        self._tune_slider.grid(row=0, column=0)

        # ── image codec mode ─────────────────────────────────────────────────
        codec_frame = ttk.LabelFrame(left, text="Image Encoding", padding=8)
        codec_frame.grid(row=3, column=0, sticky="ew", **pad)
        self._codec_frame = codec_frame

        self._codec_var = tk.StringVar(value="digital")
        ttk.Radiobutton(
            codec_frame, text="Digital", variable=self._codec_var,
            value="digital", command=self._on_codec_change,
        ).grid(row=0, column=0, padx=8)
        ttk.Radiobutton(
            codec_frame, text="Analogue", variable=self._codec_var,
            value="analogue", command=self._on_codec_change,
        ).grid(row=0, column=1, padx=8)

        # ── sink behaviour ───────────────────────────────────────────────────
        sink_frame = ttk.LabelFrame(left, text="Reconstruction Mode", padding=8)
        sink_frame.grid(row=4, column=0, sticky="ew", **pad)
        self._sink_frame = sink_frame

        self._sink_var = tk.StringVar(value="live")
        ttk.Radiobutton(
            sink_frame, text="Live", variable=self._sink_var,
            value="live", command=self._on_sink_behaviour_change,
        ).grid(row=0, column=0, padx=8)
        ttk.Radiobutton(
            sink_frame, text="Clean", variable=self._sink_var,
            value="clean", command=self._on_sink_behaviour_change,
        ).grid(row=0, column=1, padx=8)

        # ── right column ────────────────────────────────────────────────────
        right = ttk.Frame(self)
        right.grid(row=2, column=1, sticky="new")
        right.columnconfigure(0, weight=1)

        # ── image preview ────────────────────────────────────────────────────
        preview_frame = ttk.LabelFrame(right, text="Reconstructed Image", padding=8)
        preview_frame.grid(row=0, column=0, sticky="ew", **pad)
        self._preview_frame = preview_frame

        ttk.Label(preview_frame, text="Synced").grid(row=0, column=0)
        ttk.Label(preview_frame, text="Raw (no sync)").grid(row=0, column=1)

        self._preview_photo = None
        self._preview_label = ttk.Label(
            preview_frame, text="(no frame yet)", anchor="center", width=28,
        )
        self._preview_label.grid(row=1, column=0, padx=(0, 8))

        self._raw_preview_photo = None
        self._raw_preview_label = ttk.Label(
            preview_frame, text="(no frame yet)", anchor="center", width=28,
        )
        self._raw_preview_label.grid(row=1, column=1)

        # ── decoded audio export ─────────────────────────────────────────────
        export_frame = ttk.LabelFrame(right, text="Decoded Audio", padding=8)
        export_frame.grid(row=1, column=0, sticky="ew", **pad)
        self._export_frame = export_frame

        self._save_audio_btn = ttk.Button(
            export_frame, text="Save to WAV...", command=self._on_save_decoded_audio,
        )
        self._save_audio_btn.grid(row=0, column=0)

        # ── decoded binary/text output ───────────────────────────────────────
        decoded_frame = ttk.LabelFrame(right, text="Decoded Output", padding=8)
        decoded_frame.grid(row=1, column=1, sticky="ew", padx=(0, 16), pady=6)
        self._decoded_frame = decoded_frame

        ttk.Label(decoded_frame, text="Rolling (no sync)").grid(row=0, column=0)
        ttk.Label(decoded_frame, text="Clean").grid(row=0, column=1)

        self._raw_decoded_text = self._make_decoded_text_widget(decoded_frame, row=1, column=0)
        self._decoded_text = self._make_decoded_text_widget(decoded_frame, row=1, column=1)

        self._save_binary_btn = ttk.Button(
            decoded_frame, text="Save clean output to file...", command=self._on_save_decoded_binary,
        )
        self._save_binary_btn.grid(row=2, column=0, columnspan=2, pady=(6, 0))

        # ── volume ───────────────────────────────────────────────────────────
        vol_frame = ttk.LabelFrame(right, text="Monitor Volume", padding=8)
        vol_frame.grid(row=2, column=0, sticky="ew", **pad)

        self._vol_label = ttk.Label(vol_frame, text="0 dB", width=6, anchor="e")
        self._vol_label.grid(row=0, column=1, padx=(6, 0))

        self._vol_slider = ttk.Scale(
            vol_frame, from_=self._settings.volume_min_db, to=self._settings.volume_max_db,
            orient="horizontal", length=self._settings.slider_length_px,
            command=self._on_volume_change,
        )
        self._vol_slider.set(self._settings.volume_default_db)
        self._vol_slider.grid(row=0, column=0)

        # ── f0 estimator ─────────────────────────────────────────────────────
        f0_frame = ttk.LabelFrame(right, text="F0 Estimator", padding=8)
        f0_frame.grid(row=3, column=0, sticky="ew", **pad)

        self._f0_mode_var = tk.StringVar(value="Manual")
        self._f0_mode_combo = ttk.Combobox(
            f0_frame, textvariable=self._f0_mode_var,
            values=["Manual", "Autocorrelation", "FFT"], state="readonly", width=16,
        )
        self._f0_mode_combo.grid(row=0, column=0, padx=(0, 12))
        self._f0_mode_combo.bind("<<ComboboxSelected>>", self._on_f0_mode_change)

        self._quantize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f0_frame, text="Quantize Pitch", variable=self._quantize_var,
            command=self._on_quantize_change,
        ).grid(row=0, column=1)

        # ── upsample method ──────────────────────────────────────────────────
        resample_frame = ttk.LabelFrame(right, text="Symbol Upsample Method", padding=8)
        resample_frame.grid(row=6, column=0, sticky="ew", **pad)

        self._resample_method_var = tk.StringVar(value="Polyphase (resample_poly)")
        self._resample_method_combo = ttk.Combobox(
            resample_frame, textvariable=self._resample_method_var,
            values=list(self._RESAMPLE_METHOD_TO_KEY.keys()), state="readonly", width=24,
        )
        self._resample_method_combo.grid(row=0, column=0)
        self._resample_method_combo.bind("<<ComboboxSelected>>", self._on_resample_method_change)

        # ── pitch ─────────────────────────────────────────────────────────────
        pitch_frame = ttk.LabelFrame(right, text="Pitch (Hz)", padding=8)
        pitch_frame.grid(row=4, column=0, sticky="ew", **pad)
        self._pitch_frame = pitch_frame

        self._pitch_label = ttk.Label(
            pitch_frame, text=f"{self._settings.pitch_default_hz:.0f} Hz", width=7, anchor="e",
        )
        self._pitch_label.grid(row=0, column=1, padx=(6, 0))

        self._pitch_slider = ttk.Scale(
            pitch_frame, from_=self._settings.pitch_min_hz, to=self._settings.pitch_max_hz,
            orient="horizontal", length=self._settings.slider_length_px,
            command=self._on_pitch_change,
        )
        self._pitch_slider.set(self._settings.pitch_default_hz)
        self._pitch_slider.grid(row=0, column=0)

        # ── bits per symbol ───────────────────────────────────────────────────
        bits_frame = ttk.LabelFrame(right, text="Bits per Symbol", padding=8)
        bits_frame.grid(row=5, column=0, sticky="ew", **pad)

        self._bits_var = tk.StringVar(value=str(self._settings.bits_per_symbol))
        bits_combo = ttk.Combobox(
            bits_frame, textvariable=self._bits_var,
            values=[str(i) for i in range(self._settings.bits_per_symbol_min, self._settings.bits_per_symbol_max + 1)],
            state="readonly", width=6,
        )
        bits_combo.grid(row=0, column=0, padx=8)
        bits_combo.bind("<<ComboboxSelected>>", self._on_bits_change)

        self._update_kind_dependent_visibility()
        self._update_f0_mode_visibility()

    @staticmethod
    def _make_decoded_text_widget(parent: ttk.Frame, row: int, column: int) -> tk.Text:
        container = ttk.Frame(parent)
        container.grid(row=row, column=column, padx=(0, 8) if column == 0 else 0)
        text = tk.Text(container, width=24, height=8, wrap="word", state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        return text

    def _selected_device(self, var: tk.StringVar, devices: List[Tuple[int, str]]) -> Optional[int]:
        name = var.get()
        for idx, dev_name in devices:
            if dev_name == name:
                return idx
        return None

    def _apply_devices(self) -> None:
        self._engine.set_input_device(self._selected_device(self._in_device_var, self._input_devices))
        self._engine.set_output_device(self._selected_device(self._out_device_var, self._output_devices))

    def _on_device_change(self, _event=None) -> None:
        self._apply_devices()
        if self._running:
            self._engine.stop()
            self._start()

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self._apply_devices()
        try:
            self._engine.start()
        except Exception as exc:
            messagebox.showerror("Audio Error", str(exc))
            return
        self._running = True
        self._status_var.set("Running")
        self._toggle_btn.configure(text="⏹  Stop")

    def _stop(self) -> None:
        self._engine.stop()
        self._running = False
        self._status_var.set("Stopped")
        self._toggle_btn.configure(text="▶  Start")

    def _on_strategy_change(self) -> None:
        self._engine.set_strategy_kind(self._strategy_var.get())
        self._tune_slider.configure(to=self._settings.chunk_size - 1)
        self._tune_slider.set(0)
        self._tune_label.configure(text="0")

    def _on_tune_change(self, value: str) -> None:
        offset = int(float(value))
        self._engine.set_tune_offset(offset)
        self._tune_label.configure(text=str(offset))

    def _on_kind_change(self) -> None:
        self._engine.set_payload_kind(self._kind_var.get())
        self._pending_image_frame = None
        self._preview_photo = None
        self._preview_label.configure(image="", text="(no frame yet)")
        self._pending_raw_frame = None
        self._raw_preview_photo = None
        self._raw_preview_label.configure(image="", text="(no frame yet)")
        self._set_decoded_text(self._decoded_text, "(no data yet)")
        self._set_decoded_text(self._raw_decoded_text, "(no data yet)")
        self._update_kind_dependent_visibility()

    @staticmethod
    def _set_decoded_text(widget: tk.Text, text: str) -> None:
        # Tcl strings cannot contain embedded NULs; a raw decoded byte stream
        # (e.g. the binary length-prefix header) would otherwise silently
        # truncate everything inserted after the first \x00.
        text = text.replace("\x00", "�")
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _update_kind_dependent_visibility(self) -> None:
        kind = self._kind_var.get()
        is_image = kind == "image"
        is_binary_or_text = kind in ("binary", "text")
        codec_state = "normal" if kind != "audio" else "disabled"
        for frame in (self._codec_frame, self._sink_frame):
            for child in frame.winfo_children():
                try:
                    child.configure(state=codec_state)
                except tk.TclError:
                    pass
        preview_state = "normal" if is_image else "disabled"
        for child in self._preview_frame.winfo_children():
            try:
                child.configure(state=preview_state)
            except tk.TclError:
                pass
        self._save_audio_btn.configure(state="normal" if kind == "audio" else "disabled")
        self._save_binary_btn.configure(state="normal" if is_binary_or_text else "disabled")

    def _on_save_decoded_audio(self) -> None:
        file_path = filedialog.asksaveasfilename(
            title="Save decoded audio",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            self._engine.dump_decoded_audio_to_wav(file_path)
        except Exception as exc:
            messagebox.showerror("Save Audio Error", str(exc))

    def _on_decoded_data(self, data: bytes) -> None:
        self._pending_decoded_text = f"{len(data)} bytes decoded"

    def _on_raw_decoded_data(self, data: bytes) -> None:
        self._pending_raw_decoded_text = f"...{data.hex()[-400:]}"

    def _on_decoded_text(self, text: str) -> None:
        self._pending_decoded_text = text

    def _on_raw_decoded_text(self, text: str) -> None:
        self._pending_raw_decoded_text = text

    def _on_save_decoded_binary(self) -> None:
        kind = self._kind_var.get()
        file_path = filedialog.asksaveasfilename(
            title="Save decoded data",
            defaultextension=".txt" if kind == "text" else "",
            filetypes=[("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            if kind == "text":
                text = self._engine.get_latest_text()
                if text is None:
                    raise RuntimeError("No decoded text available yet.")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                self._engine.dump_decoded_bytes_to_file(file_path)
        except Exception as exc:
            messagebox.showerror("Save Data Error", str(exc))

    def _on_codec_change(self) -> None:
        mode = SerializerMode.DIGITAL if self._codec_var.get() == "digital" else SerializerMode.ANALOGUE
        self._engine.set_codec_mode(mode)

    def _on_sink_behaviour_change(self) -> None:
        behaviour = SinkBehaviour.LIVE if self._sink_var.get() == "live" else SinkBehaviour.CLEAN
        self._engine.set_sink_behaviour(behaviour)

    def _on_image_frame(self, frame) -> None:
        # Called from the audio callback thread; hand off to the Tk main loop.
        self._pending_image_frame = frame

    def _on_raw_image_frame(self, frame) -> None:
        self._pending_raw_frame = frame

    def _render_frame(self, frame, label: ttk.Label):
        pixels, width, height, channels = frame
        mode = "L" if channels == 1 else "RGB"
        try:
            image = PILImage.frombytes(mode, (width, height), bytes(pixels))
            image = image.resize(
                (self._settings.image_preview_size, self._settings.image_preview_size), PILImage.NEAREST
            )
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo, text="")
            return photo
        except Exception:
            return None

    def _poll_image(self) -> None:
        frame = self._pending_image_frame
        if frame is not None:
            photo = self._render_frame(frame, self._preview_label)
            if photo is not None:
                self._preview_photo = photo
        raw_frame = self._pending_raw_frame
        if raw_frame is not None:
            photo = self._render_frame(raw_frame, self._raw_preview_label)
            if photo is not None:
                self._raw_preview_photo = photo
        self.after(self._settings.gui_poll_interval_ms, self._poll_image)

    def _poll_decoded_text(self) -> None:
        if self._pending_decoded_text is not None:
            self._set_decoded_text(self._decoded_text, self._pending_decoded_text)
            self._pending_decoded_text = None
        if self._pending_raw_decoded_text is not None:
            self._set_decoded_text(self._raw_decoded_text, self._pending_raw_decoded_text)
            self._pending_raw_decoded_text = None
        self.after(self._settings.gui_poll_interval_ms, self._poll_decoded_text)

    def _on_bits_change(self, _event=None) -> None:
        try:
            self._engine.set_bits_per_symbol(int(self._bits_var.get()))
        except Exception as exc:
            messagebox.showerror("Bits per Symbol Error", str(exc))
            self._bits_var.set(str(self._settings.bits_per_symbol))

    def _on_volume_change(self, value: str) -> None:
        db = float(value)
        self._settings.volume_default_db = db
        gain = 0.0 if db <= self._settings.volume_min_db else 10 ** (db / 20.0)
        self._engine.set_volume(gain)
        label = "−∞ dB" if db <= self._settings.volume_min_db else f"{db:.0f} dB"
        self._vol_label.configure(text=label)

    def _on_pitch_change(self, value: str) -> None:
        f0 = float(value)
        self._settings.pitch_default_hz = f0
        self._engine.set_f0(f0)
        self._pitch_label.configure(text=f"{f0:.2f} Hz")

    _F0_MODE_TO_KEY = {"Manual": "manual", "Autocorrelation": "autocorr", "FFT": "fft"}

    def _on_f0_mode_change(self, _event=None) -> None:
        mode = self._F0_MODE_TO_KEY[self._f0_mode_var.get()]
        self._engine.set_f0_estimator_mode(mode)
        self._update_f0_mode_visibility()

    def _on_quantize_change(self) -> None:
        self._engine.set_pitch_quantize(self._quantize_var.get())

    def _on_resample_method_change(self, _event=None) -> None:
        method = self._RESAMPLE_METHOD_TO_KEY[self._resample_method_var.get()]
        self._engine.set_resample_method(method)

    def _update_f0_mode_visibility(self) -> None:
        is_manual = self._f0_mode_var.get() == "Manual"
        state = "normal" if is_manual else "disabled"
        self._pitch_slider.configure(state=state)

    def _poll_estimated_f0(self) -> None:
        is_manual = self._f0_mode_var.get() == "Manual"
        if not is_manual:
            f0 = self._engine.get_estimated_f0()
            if f0 > 0.0:
                self._pitch_slider.set(min(max(f0, self._settings.pitch_min_hz), self._settings.pitch_max_hz))
                self._pitch_label.configure(text=f"{f0:.2f} Hz")

        drop_run = self._engine.get_drop_run()
        if self._engine.is_gated() or drop_run > 0:
            self._signal_var.set(f"SIGNAL WEAK — drop {drop_run}/{self._settings.drop_tolerance_chunks}")
        elif not is_manual:
            self._signal_var.set(f"confidence {self._engine.get_confidence():.2f}")
        else:
            self._signal_var.set("")
        self.after(self._settings.gui_poll_interval_ms, self._poll_estimated_f0)

    def _on_close(self) -> None:
        self._stop()
        self.destroy()


if __name__ == "__main__":
    app = DecoderApp()
    app.mainloop()
