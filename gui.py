import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from AdditiveWaveGenerator import AdditiveWaveGenerator
from AudioChunk import AudioChunk
from Decoder import Decoder, TwoSplitDecodingStrategy
from Deserializer import AudioDeserializer
from Encoder import Encoder, TwoSplitEncodingStrategy
from Framing import FramingSyncController
from Payload import AudioPayload
from Serializer import AudioSerializer
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import AudioSink, SinkBehaviour


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

        payload = AudioPayload()
        payload.load_from_file(settings.modulator_wav_path)

        serializer = AudioSerializer(settings, SerializerMode.DIGITAL)
        enc_strategy = TwoSplitEncodingStrategy(
            settings, _make_harmonic_generator(settings), serializer
        )
        enc_strategy.load_payload(payload)
        self._encoder = Encoder(enc_strategy)

        self._dec_strategy = TwoSplitDecodingStrategy(settings, _make_harmonic_generator(settings))
        sink = AudioSink(FramingSyncController(), SinkBehaviour.LIVE)
        deserializer = AudioDeserializer(settings, sink, SerializerMode.DIGITAL)
        self._decoder = Decoder(settings, self._dec_strategy, deserializer)

        self.set_f0(440.0)

    def set_volume(self, vol: float) -> None:
        self._volume = float(vol)

    def set_source(self, source: str) -> None:
        self._source = source

    def set_f0(self, f0: float) -> None:
        self._encoder.set_f0(f0)
        self._dec_strategy.set_f0(f0)

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

    def __init__(self):
        super().__init__()
        self.title("SteganographySynthFlow")
        self.resizable(False, False)

        settings = Settings()
        self._engine = AudioEngine(settings)
        self._running = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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

        # ── output source ────────────────────────────────────────────────────
        src_frame = ttk.LabelFrame(self, text="Output Source", padding=8)
        src_frame.grid(row=2, column=0, sticky="ew", **pad)

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
        vol_frame.grid(row=3, column=0, sticky="ew", **pad)

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
        pitch_frame.grid(row=4, column=0, sticky="ew", **pad)

        self._pitch_label = ttk.Label(pitch_frame, text="440 Hz", width=7, anchor="e")
        self._pitch_label.grid(row=0, column=1, padx=(6, 0))

        self._pitch_slider = ttk.Scale(
            pitch_frame, from_=self._F0_MIN, to=self._F0_MAX, orient="horizontal", length=280,
            command=self._on_pitch_change,
        )
        self._pitch_slider.set(440)
        self._pitch_slider.grid(row=0, column=0)

        # ── bottom padding ────────────────────────────────────────────────────
        ttk.Frame(self).grid(row=5, pady=6)

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
