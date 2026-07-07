import os
import threading
import tkinter as tk
from typing import Callable, Optional
from tkinter import filedialog, ttk, messagebox

import numpy as np
from PIL import Image as PILImage, ImageTk

try:
    import sounddevice as sd
except ImportError:
    sd = None

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Decoder import Decoder, TwoSplitDecodingStrategy
from Deserializer import AudioDeserializer, ImageDeserializer
from Encoder import Encoder, TwoSplitEncodingStrategy
from Framing import FramingSyncController
from Payload import AudioPayload, ImagePayload
from Payload.pixel_codec import make_pixel_codec
from Serializer import AudioSerializer, ImageSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import AudioSink, ImageSink, RawImageSink, SinkBehaviour, SinkTee


def _make_harmonic_generator(settings: Settings) -> AdditiveWaveGenerator:
    gen = AdditiveWaveGenerator(settings)
    gen.set_omegas([float(i + 1) for i in range(settings.total_harmonics)])
    gen.set_phases([0.0] * settings.total_harmonics)
    gen.set_amps([settings.base_amplitude / (i + 1) for i in range(settings.total_harmonics)])
    return gen


class AudioEngine:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._volume: float = 1.0
        self._source: str = "encoder"
        self._stream = None
        self._lock = threading.Lock()

        self._payload_kind: str = "audio"
        self._codec_mode: SerializerMode = SerializerMode.DIGITAL
        self._sink_behaviour: SinkBehaviour = SinkBehaviour.LIVE
        self._on_image: Optional[Callable] = None
        self._image_sink: Optional[ImageSink] = None
        self._on_raw_image: Optional[Callable] = None
        self._raw_image_sink: Optional[RawImageSink] = None
        self._f0: float = 0.0

        self._audio_payload = AudioPayload()
        self._audio_payload.load_from_file(settings.modulator_wav_path)

        self._image_path = settings.image_path
        self._image_codec = make_pixel_codec(self._codec_mode, settings)
        self._image_payload = ImagePayload(settings, self._image_codec)
        self._image_payload.load_from_file(self._image_path)

        self._dec_strategy = TwoSplitDecodingStrategy(settings, _make_harmonic_generator(settings))

        self._encoding_strategy = self._build_audio_encoding_strategy()
        self._encoder = Encoder(self._encoding_strategy)

        self._decoder: Optional[Decoder] = None
        self._rebuild_decode_chain()

        self.set_f0(400.0)

    # ── encoding strategy construction ───────────────────────────────────────
    def _build_audio_encoding_strategy(self) -> TwoSplitEncodingStrategy:
        serializer = AudioSerializer(self._settings, SerializerMode.DIGITAL)
        strategy = TwoSplitEncodingStrategy(
            self._settings, _make_harmonic_generator(self._settings), serializer
        )
        strategy.load_payload(self._audio_payload)
        return strategy

    def _build_image_encoding_strategy(self) -> TwoSplitEncodingStrategy:
        serializer = ImageSerializer(self._settings, self._codec_mode)
        strategy = TwoSplitEncodingStrategy(
            self._settings, _make_harmonic_generator(self._settings), serializer
        )
        strategy.load_payload(self._image_payload)
        return strategy

    def _rebuild_decode_chain(self) -> None:
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
            deserializer = ImageDeserializer(
                self._settings, SinkTee(sink, raw_sink), self._codec_mode
            )
        else:
            self._image_sink = None
            self._raw_image_sink = None
            sink = AudioSink(FramingSyncController(), SinkBehaviour.LIVE)
            deserializer = AudioDeserializer(self._settings, sink, SerializerMode.DIGITAL)

        self._decoder = Decoder(self._settings, self._dec_strategy, deserializer)

    # ── public controls ──────────────────────────────────────────────────────
    def set_volume(self, vol: float) -> None:
        self._volume = float(vol)

    def set_source(self, source: str) -> None:
        self._source = source

    def get_payload_kind(self) -> str:
        return self._payload_kind

    def set_payload_kind(self, kind: str) -> None:
        if kind not in ("audio", "image"):
            raise ValueError(f"Unknown payload kind: {kind}")
        with self._lock:
            self._payload_kind = kind
            self._encoding_strategy = (
                self._build_audio_encoding_strategy() if kind == "audio"
                else self._build_image_encoding_strategy()
            )
            self._encoding_strategy.set_f0(self._f0)
            self._encoder.set_encoding_strategy(self._encoding_strategy)
            self._dec_strategy.reconfigure()
            self._rebuild_decode_chain()

    def set_codec_mode(self, mode: SerializerMode) -> None:
        with self._lock:
            self._codec_mode = mode
            self._image_codec = make_pixel_codec(mode, self._settings)
            payload = ImagePayload(self._settings, self._image_codec)
            payload.load_from_file(self._image_path)
            self._image_payload = payload
            if self._payload_kind == "image":
                self._encoding_strategy = self._build_image_encoding_strategy()
                self._encoding_strategy.set_f0(self._f0)
                self._encoder.set_encoding_strategy(self._encoding_strategy)
                self._dec_strategy.reconfigure()
                self._rebuild_decode_chain()

    def set_sink_behaviour(self, behaviour: SinkBehaviour) -> None:
        with self._lock:
            self._sink_behaviour = behaviour
            if self._payload_kind == "image":
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

    def get_latest_image(self):
        with self._lock:
            return self._image_sink.get_latest_image() if self._image_sink is not None else None

    def load_payload_file(self, file_path: str) -> None:
        with self._lock:
            if self._payload_kind == "image":
                self._image_path = file_path
                payload = ImagePayload(self._settings, self._image_codec)
                payload.load_from_file(file_path)
                self._image_payload = payload
                self._encoding_strategy = self._build_image_encoding_strategy()
            else:
                payload = AudioPayload()
                payload.load_from_file(file_path)
                self._audio_payload = payload
                self._encoding_strategy = self._build_audio_encoding_strategy()
            self._encoding_strategy.set_f0(self._f0)
            self._encoder.set_encoding_strategy(self._encoding_strategy)
            self._dec_strategy.reconfigure()

    def get_position_fraction(self) -> float:
        with self._lock:
            return self._encoding_strategy.get_position_fraction()

    def set_position_fraction(self, fraction: float) -> None:
        with self._lock:
            self._encoding_strategy.set_position_fraction(fraction)

    def set_f0(self, f0: float) -> None:
        self._f0 = float(f0)
        self._encoder.set_f0(f0)
        self._dec_strategy.set_f0(f0)

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        with self._lock:
            self._settings.set_bits_per_symbol(int(bits_per_symbol))
            self._image_codec = make_pixel_codec(self._codec_mode, self._settings)
            payload = ImagePayload(self._settings, self._image_codec)
            payload.load_from_file(self._image_path)
            self._image_payload = payload
            self._encoding_strategy = (
                self._build_audio_encoding_strategy() if self._payload_kind == "audio"
                else self._build_image_encoding_strategy()
            )
            self._encoding_strategy.set_f0(self._f0)
            self._encoder.set_encoding_strategy(self._encoding_strategy)
            self._dec_strategy.reconfigure()
            self._rebuild_decode_chain()

    def _callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            enc_chunk = self._encoder.process(frames)
            enc_samples = enc_chunk.get_samples()

            dec_chunk = self._decoder.process(AudioChunk(enc_samples), frames)
            dec_samples = dec_chunk.get_samples()

            samples = enc_samples if self._source == "encoder" else dec_samples
            arr = np.array(samples, dtype=np.float32) * self._volume
            outdata[:, 0] = arr

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed.\nRun: pip install sounddevice")
        block_size = self._settings.audio_driver_polling_rate
        self._stream = sd.OutputStream(
            samplerate=self._settings.fs_out,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class App(tk.Tk):
    _F0_MIN = 100.0
    _F0_MAX = 2000.0
    _IMAGE_PREVIEW_SIZE = 200

    def __init__(self):
        super().__init__()
        self.title("SteganographySynthFlow")
        self.resizable(False, False)

        settings = Settings()
        self._engine = AudioEngine(settings)
        self._engine.set_on_image(self._on_image_frame)
        self._engine.set_on_raw_image(self._on_raw_image_frame)
        self._running = False
        self._pending_image_frame = None
        self._pending_raw_frame = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_position()
        self._poll_image()

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 6}

        # ── status ──────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Stopped")
        status_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        status_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 0))
        ttk.Label(status_frame, text="Status:").pack(side="left", padx=6, pady=4)
        ttk.Label(status_frame, textvariable=self._status_var, font=("", 10, "bold")).pack(
            side="left", pady=4
        )
        self.columnconfigure(0, weight=1)

        # ── start / stop ─────────────────────────────────────────────────────
        self._toggle_btn = ttk.Button(self, text="▶  Start", command=self._toggle, width=14)
        self._toggle_btn.grid(row=1, column=0, **pad)

        # ── payload kind ──────────────────────────────────────────────────────
        kind_frame = ttk.LabelFrame(self, text="Payload Type", padding=8)
        kind_frame.grid(row=2, column=0, sticky="ew", **pad)

        self._kind_var = tk.StringVar(value="audio")
        ttk.Radiobutton(
            kind_frame, text="Audio", variable=self._kind_var,
            value="audio", command=self._on_kind_change,
        ).grid(row=0, column=0, padx=8)
        ttk.Radiobutton(
            kind_frame, text="Image", variable=self._kind_var,
            value="image", command=self._on_kind_change,
        ).grid(row=0, column=1, padx=8)

        # ── payload file + position ──────────────────────────────────────────
        payload_frame = ttk.LabelFrame(self, text="Payload", padding=8)
        payload_frame.grid(row=3, column=0, sticky="ew", **pad)
        payload_frame.columnconfigure(0, weight=1)
        self._payload_frame = payload_frame

        self._payload_var = tk.StringVar(
            value=os.path.basename(self._engine._settings.modulator_wav_path)
        )
        ttk.Label(payload_frame, textvariable=self._payload_var, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(payload_frame, text="Browse...", command=self._on_pick_payload).grid(
            row=0, column=1
        )

        self._payload_dragging = False
        self._position_slider = ttk.Scale(
            payload_frame, from_=0, to=1000, orient="horizontal", length=280,
        )
        self._position_slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self._position_slider.bind("<ButtonPress-1>", self._on_position_drag_start)
        self._position_slider.bind("<ButtonRelease-1>", self._on_position_drag_end)

        # ── image codec mode (digital / analogue) ────────────────────────────
        codec_frame = ttk.LabelFrame(self, text="Image Encoding", padding=8)
        codec_frame.grid(row=4, column=0, sticky="ew", **pad)
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

        # ── image sink behaviour (live / clean) ──────────────────────────────
        sink_frame = ttk.LabelFrame(self, text="Reconstruction Mode", padding=8)
        sink_frame.grid(row=5, column=0, sticky="ew", **pad)
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

        # ── reconstructed image preview ──────────────────────────────────────
        preview_frame = ttk.LabelFrame(self, text="Reconstructed Image", padding=8)
        preview_frame.grid(row=6, column=0, sticky="ew", **pad)
        self._preview_frame = preview_frame

        ttk.Label(preview_frame, text="Synced").grid(row=0, column=0)
        ttk.Label(preview_frame, text="Raw (no sync)").grid(row=0, column=1)

        self._preview_photo = None
        self._preview_label = ttk.Label(
            preview_frame, text="(no frame yet)", anchor="center",
            width=28,
        )
        self._preview_label.grid(row=1, column=0, padx=(0, 8))

        self._raw_preview_photo = None
        self._raw_preview_label = ttk.Label(
            preview_frame, text="(no frame yet)", anchor="center",
            width=28,
        )
        self._raw_preview_label.grid(row=1, column=1)

        # ── output source ────────────────────────────────────────────────────
        src_frame = ttk.LabelFrame(self, text="Output Source", padding=8)
        src_frame.grid(row=7, column=0, sticky="ew", **pad)

        self._source_var = tk.StringVar(value="encoder")
        ttk.Radiobutton(
            src_frame, text="Encoder", variable=self._source_var,
            value="encoder", command=self._on_source_change,
        ).grid(row=0, column=0, padx=8)
        ttk.Radiobutton(
            src_frame, text="Decoder", variable=self._source_var,
            value="decoder", command=self._on_source_change,
        ).grid(row=0, column=1, padx=8)

        # ── volume ───────────────────────────────────────────────────────────
        vol_frame = ttk.LabelFrame(self, text="Volume", padding=8)
        vol_frame.grid(row=8, column=0, sticky="ew", **pad)

        self._vol_label = ttk.Label(vol_frame, text="0 dB", width=6, anchor="e")
        self._vol_label.grid(row=0, column=1, padx=(6, 0))

        # Slider position is in dB: -60 (min) to 0 (max). Position 0 = silence.
        self._vol_slider = ttk.Scale(
            vol_frame, from_=-60, to=0, orient="horizontal", length=280,
            command=self._on_volume_change,
        )
        self._vol_slider.set(-40)
        self._vol_slider.grid(row=0, column=0)

        # ── pitch ─────────────────────────────────────────────────────────────
        pitch_frame = ttk.LabelFrame(self, text="Pitch (Hz)", padding=8)
        pitch_frame.grid(row=9, column=0, sticky="ew", **pad)

        self._pitch_label = ttk.Label(pitch_frame, text="400 Hz", width=7, anchor="e")
        self._pitch_label.grid(row=0, column=1, padx=(6, 0))

        self._pitch_slider = ttk.Scale(
            pitch_frame, from_=self._F0_MIN, to=self._F0_MAX, orient="horizontal", length=280,
            command=self._on_pitch_change,
        )
        self._pitch_slider.set(400)
        self._pitch_slider.grid(row=0, column=0)

        # ── bits per symbol ───────────────────────────────────────────────────
        bits_frame = ttk.LabelFrame(self, text="Bits per Symbol", padding=8)
        bits_frame.grid(row=10, column=0, sticky="ew", **pad)

        self._bits_var = tk.StringVar(value=str(self._engine._settings.bits_per_symbol))
        bits_combo = ttk.Combobox(
            bits_frame, textvariable=self._bits_var,
            values=[str(i) for i in range(1, 9)], state="readonly", width=6,
        )
        bits_combo.grid(row=0, column=0, padx=8)
        bits_combo.bind("<<ComboboxSelected>>", self._on_bits_change)

        # ── bottom padding ────────────────────────────────────────────────────
        ttk.Frame(self).grid(row=11, pady=6)

        self._update_kind_dependent_visibility()

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        try:
            self._engine.start()
        except RuntimeError as exc:
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

    def _on_pick_payload(self) -> None:
        if self._kind_var.get() == "image":
            file_path = filedialog.askopenfilename(
                title="Select image payload",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                    ("All files", "*.*"),
                ],
            )
        else:
            file_path = filedialog.askopenfilename(
                title="Select audio payload",
                filetypes=[
                    ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                    ("All files", "*.*"),
                ],
            )
        if not file_path:
            return
        try:
            self._engine.load_payload_file(file_path)
        except Exception as exc:
            messagebox.showerror("Payload Error", str(exc))
            return
        self._payload_var.set(os.path.basename(file_path))

    def _on_position_drag_start(self, _event) -> None:
        self._payload_dragging = True

    def _on_position_drag_end(self, _event) -> None:
        fraction = self._position_slider.get() / 1000.0
        self._engine.set_position_fraction(fraction)
        self._payload_dragging = False

    def _poll_position(self) -> None:
        if not self._payload_dragging:
            fraction = self._engine.get_position_fraction()
            self._position_slider.set(fraction * 1000.0)
        self.after(100, self._poll_position)

    def _on_kind_change(self) -> None:
        kind = self._kind_var.get()
        self._engine.set_payload_kind(kind)
        default_name = (
            os.path.basename(self._engine._image_path) if kind == "image"
            else os.path.basename(self._engine._settings.modulator_wav_path)
        )
        self._payload_var.set(default_name)
        self._pending_image_frame = None
        self._preview_photo = None
        self._preview_label.configure(image="", text="(no frame yet)")
        self._pending_raw_frame = None
        self._raw_preview_photo = None
        self._raw_preview_label.configure(image="", text="(no frame yet)")
        self._update_kind_dependent_visibility()

    def _update_kind_dependent_visibility(self) -> None:
        is_image = self._kind_var.get() == "image"
        state = "normal" if is_image else "disabled"
        for frame in (self._codec_frame, self._sink_frame, self._preview_frame):
            for child in frame.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass

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
                (self._IMAGE_PREVIEW_SIZE, self._IMAGE_PREVIEW_SIZE), PILImage.NEAREST
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
        self.after(100, self._poll_image)

    def _on_bits_change(self, _event=None) -> None:
        try:
            self._engine.set_bits_per_symbol(int(self._bits_var.get()))
        except Exception as exc:
            messagebox.showerror("Bits per Symbol Error", str(exc))
            self._bits_var.set(str(self._engine._settings.bits_per_symbol))

    def _on_source_change(self) -> None:
        self._engine.set_source(self._source_var.get())

    def _on_volume_change(self, value: str) -> None:
        db = float(value)
        gain = 0.0 if db <= -60 else 10 ** (db / 20.0)
        self._engine.set_volume(gain)
        label = "−∞ dB" if db <= -60 else f"{db:.0f} dB"
        self._vol_label.configure(text=label)

    def _on_pitch_change(self, value: str) -> None:
        f0 = float(value)
        self._engine.set_f0(f0)
        self._pitch_label.configure(text=f"{int(f0)} Hz")

    def _on_close(self) -> None:
        self._stop()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
