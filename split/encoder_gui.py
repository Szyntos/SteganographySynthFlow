import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from typing import List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # Settings uses paths relative to the project root

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from AdditiveWaveGenerator import AdditiveWaveGenerator
from Encoder import Encoder, TwoSplitEncodingStrategy
from Payload import AudioPayload, ImagePayload
from Payload.pixel_codec import make_pixel_codec
from Serializer import AudioSerializer, ImageSerializer
from SerializerMode import SerializerMode
from Settings import Settings


def _make_harmonic_generator(settings: Settings) -> AdditiveWaveGenerator:
    gen = AdditiveWaveGenerator(settings)
    gen.set_omegas([float(i + 1) for i in range(settings.total_harmonics)])
    gen.set_phases([0.0] * settings.total_harmonics)
    gen.set_amps([settings.base_amplitude / (i + 1) for i in range(settings.total_harmonics)])
    return gen


def list_output_devices() -> List[Tuple[int, str]]:
    if sd is None:
        return []
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            devices.append((idx, f"[{idx}] {dev['name']}"))
    return devices


class EncoderEngine:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._volume: float = 1.0
        self._stream = None
        self._lock = threading.Lock()
        self._output_device: Optional[int] = None

        self._payload_kind: str = "audio"
        self._codec_mode: SerializerMode = SerializerMode.DIGITAL
        self._f0: float = 400.0

        self._audio_payload = AudioPayload()
        self._audio_payload.load_from_file(settings.modulator_wav_path)

        self._image_path = settings.image_path
        self._image_codec = make_pixel_codec(self._codec_mode, settings)
        self._image_payload = ImagePayload(settings, self._image_codec)
        self._image_payload.load_from_file(self._image_path)

        self._encoding_strategy = self._build_audio_encoding_strategy()
        self._encoder = Encoder(self._encoding_strategy)
        self.set_f0(self._f0)

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

    # ── public controls ──────────────────────────────────────────────────────
    def set_volume(self, vol: float) -> None:
        self._volume = float(vol)

    def set_output_device(self, device: Optional[int]) -> None:
        self._output_device = device

    def is_running(self) -> bool:
        return self._stream is not None

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

    def get_position_fraction(self) -> float:
        with self._lock:
            return self._encoding_strategy.get_position_fraction()

    def set_position_fraction(self, fraction: float) -> None:
        with self._lock:
            self._encoding_strategy.set_position_fraction(fraction)

    def set_f0(self, f0: float) -> None:
        self._f0 = float(f0)
        self._encoder.set_f0(f0)

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

    def _callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            enc_chunk = self._encoder.process(frames)
            arr = np.array(enc_chunk.get_samples(), dtype=np.float32) * self._volume
            outdata[:, 0] = arr

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed.\nRun: pip install sounddevice")
        self._stream = sd.OutputStream(
            samplerate=self._settings.fs_out,
            blocksize=self._settings.audio_driver_polling_rate,
            channels=1,
            dtype="float32",
            device=self._output_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class EncoderApp(tk.Tk):
    _F0_MIN = 100.0
    _F0_MAX = 2000.0

    def __init__(self):
        super().__init__()
        self.title("SSF Encoder")
        self.resizable(False, False)

        settings = Settings()
        self._engine = EncoderEngine(settings)
        self._running = False
        self._devices = list_output_devices()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_position()

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

        # ── output device ────────────────────────────────────────────────────
        dev_frame = ttk.LabelFrame(self, text="Output Device", padding=8)
        dev_frame.grid(row=2, column=0, sticky="ew", **pad)
        dev_frame.columnconfigure(0, weight=1)

        names = ["(default)"] + [name for _, name in self._devices]
        self._device_var = tk.StringVar(value=names[0])
        self._device_combo = ttk.Combobox(
            dev_frame, textvariable=self._device_var, values=names, state="readonly", width=42
        )
        self._device_combo.grid(row=0, column=0, sticky="ew")
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_change)

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

        # ── payload file + position ──────────────────────────────────────────
        payload_frame = ttk.LabelFrame(self, text="Payload", padding=8)
        payload_frame.grid(row=4, column=0, sticky="ew", **pad)
        payload_frame.columnconfigure(0, weight=1)

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

        # ── volume ───────────────────────────────────────────────────────────
        vol_frame = ttk.LabelFrame(self, text="Volume", padding=8)
        vol_frame.grid(row=6, column=0, sticky="ew", **pad)

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
        pitch_frame.grid(row=7, column=0, sticky="ew", **pad)

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
        bits_frame.grid(row=8, column=0, sticky="ew", **pad)

        self._bits_var = tk.StringVar(value=str(self._engine._settings.bits_per_symbol))
        bits_combo = ttk.Combobox(
            bits_frame, textvariable=self._bits_var,
            values=[str(i) for i in range(1, 9)], state="readonly", width=6,
        )
        bits_combo.grid(row=0, column=0, padx=8)
        bits_combo.bind("<<ComboboxSelected>>", self._on_bits_change)

        ttk.Frame(self).grid(row=9, pady=6)
        self._update_kind_dependent_visibility()

    def _selected_device(self) -> Optional[int]:
        name = self._device_var.get()
        for idx, dev_name in self._devices:
            if dev_name == name:
                return idx
        return None

    def _on_device_change(self, _event=None) -> None:
        self._engine.set_output_device(self._selected_device())
        if self._running:
            self._engine.stop()
            self._start()

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self._engine.set_output_device(self._selected_device())
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
        self._update_kind_dependent_visibility()

    def _update_kind_dependent_visibility(self) -> None:
        state = "normal" if self._kind_var.get() == "image" else "disabled"
        for child in self._codec_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _on_codec_change(self) -> None:
        mode = SerializerMode.DIGITAL if self._codec_var.get() == "digital" else SerializerMode.ANALOGUE
        self._engine.set_codec_mode(mode)

    def _on_bits_change(self, _event=None) -> None:
        try:
            self._engine.set_bits_per_symbol(int(self._bits_var.get()))
        except Exception as exc:
            messagebox.showerror("Bits per Symbol Error", str(exc))
            self._bits_var.set(str(self._engine._settings.bits_per_symbol))

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
    app = EncoderApp()
    app.mainloop()
