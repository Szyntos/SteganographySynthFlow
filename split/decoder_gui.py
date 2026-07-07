import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
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

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Decoder import Decoder, TwoSplitDecodingStrategy
from Deserializer import AudioDeserializer, ImageDeserializer
from Framing import FramingSyncController
from Payload.pixel_codec import make_pixel_codec
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import AudioSink, ImageSink, RawImageSink, SinkBehaviour, SinkTee


def _make_harmonic_generator(settings: Settings) -> AdditiveWaveGenerator:
    gen = AdditiveWaveGenerator(settings)
    gen.set_omegas([float(i + 1) for i in range(settings.total_harmonics)])
    gen.set_phases([0.0] * settings.total_harmonics)
    gen.set_amps([settings.base_amplitude / (i + 1) for i in range(settings.total_harmonics)])
    return gen


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
        self._codec_mode: SerializerMode = SerializerMode.DIGITAL
        self._sink_behaviour: SinkBehaviour = SinkBehaviour.LIVE
        self._on_image: Optional[Callable] = None
        self._image_sink: Optional[ImageSink] = None
        self._on_raw_image: Optional[Callable] = None
        self._raw_image_sink: Optional[RawImageSink] = None
        self._image_codec = make_pixel_codec(self._codec_mode, settings)

        # FIFO tuning: how many input samples still to drop so the decode
        # window slides to the requested offset within a chunk.
        self._tune_offset: int = 0
        self._pending_skip: int = 0

        self._dec_strategy = TwoSplitDecodingStrategy(settings, _make_harmonic_generator(settings))
        self._decoder: Optional[Decoder] = None
        self._rebuild_decode_chain()
        self.set_f0(400.0)

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

    def set_input_device(self, device: Optional[int]) -> None:
        self._input_device = device

    def set_output_device(self, device: Optional[int]) -> None:
        self._output_device = device

    def is_running(self) -> bool:
        return self._stream is not None

    def set_payload_kind(self, kind: str) -> None:
        if kind not in ("audio", "image"):
            raise ValueError(f"Unknown payload kind: {kind}")
        with self._lock:
            self._payload_kind = kind
            self._dec_strategy.reconfigure()
            self._rebuild_decode_chain()

    def set_codec_mode(self, mode: SerializerMode) -> None:
        with self._lock:
            self._codec_mode = mode
            self._image_codec = make_pixel_codec(mode, self._settings)
            if self._payload_kind == "image":
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

    def set_f0(self, f0: float) -> None:
        self._dec_strategy.set_f0(float(f0))

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        with self._lock:
            self._settings.set_bits_per_symbol(int(bits_per_symbol))
            self._image_codec = make_pixel_codec(self._codec_mode, self._settings)
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

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            samples = indata[:, 0].astype(np.float32)

            if self._pending_skip > 0:
                drop = min(self._pending_skip, len(samples))
                samples = samples[drop:]
                self._pending_skip -= drop

            dec_chunk = self._decoder.process(AudioChunk(samples.tolist()), frames)
            arr = np.array(dec_chunk.get_samples(), dtype=np.float32) * self._volume
            outdata[:, 0] = arr

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed.\nRun: pip install sounddevice")
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
    _F0_MIN = 100.0
    _F0_MAX = 2000.0
    _IMAGE_PREVIEW_SIZE = 200

    def __init__(self):
        super().__init__()
        self.title("SSF Decoder")
        self.resizable(False, False)

        settings = Settings()
        self._settings = settings
        self._engine = DecoderEngine(settings)
        self._engine.set_on_image(self._on_image_frame)
        self._engine.set_on_raw_image(self._on_raw_image_frame)
        self._running = False
        self._pending_image_frame = None
        self._pending_raw_frame = None
        self._input_devices = list_devices("input")
        self._output_devices = list_devices("output")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_image()

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 6}

        self._status_var = tk.StringVar(value="Stopped")
        status_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        status_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 0))
        ttk.Label(status_frame, text="Status:").pack(side="left", padx=6, pady=4)
        ttk.Label(status_frame, textvariable=self._status_var, font=("", 10, "bold")).pack(
            side="left", pady=4
        )
        self.columnconfigure(0, weight=1)

        self._toggle_btn = ttk.Button(self, text="▶  Start", command=self._toggle, width=14)
        self._toggle_btn.grid(row=1, column=0, **pad)

        # ── devices ──────────────────────────────────────────────────────────
        dev_frame = ttk.LabelFrame(self, text="Audio Devices", padding=8)
        dev_frame.grid(row=2, column=0, sticky="ew", **pad)
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

        # ── payload kind ──────────────────────────────────────────────────────
        kind_frame = ttk.LabelFrame(self, text="Payload Type", padding=8)
        kind_frame.grid(row=3, column=0, sticky="ew", **pad)

        self._kind_var = tk.StringVar(value="audio")
        ttk.Radiobutton(
            kind_frame, text="Audio", variable=self._kind_var,
            value="audio", command=self._on_kind_change,
        ).grid(row=0, column=0, padx=8)
        ttk.Radiobutton(
            kind_frame, text="Image", variable=self._kind_var,
            value="image", command=self._on_kind_change,
        ).grid(row=0, column=1, padx=8)

        # ── fifo tuning ──────────────────────────────────────────────────────
        tune_frame = ttk.LabelFrame(self, text="Window Tuning (samples)", padding=8)
        tune_frame.grid(row=4, column=0, sticky="ew", **pad)

        self._tune_label = ttk.Label(tune_frame, text="0", width=6, anchor="e")
        self._tune_label.grid(row=0, column=1, padx=(6, 0))

        self._tune_slider = ttk.Scale(
            tune_frame, from_=0, to=self._settings.chunk_size - 1,
            orient="horizontal", length=280, command=self._on_tune_change,
        )
        self._tune_slider.set(0)
        self._tune_slider.grid(row=0, column=0)

        # ── image codec mode ─────────────────────────────────────────────────
        codec_frame = ttk.LabelFrame(self, text="Image Encoding", padding=8)
        codec_frame.grid(row=5, column=0, sticky="ew", **pad)
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
        sink_frame = ttk.LabelFrame(self, text="Reconstruction Mode", padding=8)
        sink_frame.grid(row=6, column=0, sticky="ew", **pad)
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

        # ── image preview ────────────────────────────────────────────────────
        preview_frame = ttk.LabelFrame(self, text="Reconstructed Image", padding=8)
        preview_frame.grid(row=7, column=0, sticky="ew", **pad)
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

        # ── volume ───────────────────────────────────────────────────────────
        vol_frame = ttk.LabelFrame(self, text="Monitor Volume", padding=8)
        vol_frame.grid(row=8, column=0, sticky="ew", **pad)

        self._vol_label = ttk.Label(vol_frame, text="0 dB", width=6, anchor="e")
        self._vol_label.grid(row=0, column=1, padx=(6, 0))

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

        self._bits_var = tk.StringVar(value=str(self._settings.bits_per_symbol))
        bits_combo = ttk.Combobox(
            bits_frame, textvariable=self._bits_var,
            values=[str(i) for i in range(1, 9)], state="readonly", width=6,
        )
        bits_combo.grid(row=0, column=0, padx=8)
        bits_combo.bind("<<ComboboxSelected>>", self._on_bits_change)

        ttk.Frame(self).grid(row=11, pady=6)
        self._update_kind_dependent_visibility()

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
            self._bits_var.set(str(self._settings.bits_per_symbol))

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
    app = DecoderApp()
    app.mainloop()
