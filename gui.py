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

from DecoderDSP import DecoderDSP
from EncoderDSP import EncoderDSP
from MidiFilePlayer import MidiFilePlayer
from NoteState import NoteState, midi_note_to_hz
from PianoKeyboard import PianoKeyboard
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import SinkBehaviour


class AudioEngine:
    """Drives the encoder/decoder DSP pipelines against a live audio device.

    Assembly of strategies, payloads, codecs and sinks lives in EncoderDSP/
    DecoderDSP; this class only owns the sounddevice stream, the output
    routing (encoder vs. decoder), and keeps both DSP objects' shared knobs
    (strategy kind, payload kind, codec mode, bits/symbol, f0) in sync.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._volume: float = 1.0
        self._source: str = "encoder"
        self._stream = None
        self._lock = threading.Lock()

        self._enc = EncoderDSP(settings)
        self._dec = DecoderDSP(settings)
        self.set_f0(settings.pitch_default_hz)

        self._note_state = NoteState()
        self._midi_file_player = MidiFilePlayer(self._note_state)

    def get_midi_file_player(self) -> MidiFilePlayer:
        return self._midi_file_player

    def get_active_note(self) -> Optional[int]:
        note = self._note_state.current_note_or(-1)
        return note if note >= 0 else None

    def set_f0_estimator_mode(self, mode: str) -> None:
        with self._lock:
            self._dec.set_f0_estimator_mode(mode)

    def set_pitch_quantize(self, enabled: bool) -> None:
        self._dec.set_pitch_quantize(enabled)

    def get_estimated_f0(self) -> float:
        return self._dec.get_estimated_f0()

    def get_confidence(self) -> float:
        return self._dec.get_confidence()

    def get_drop_run(self) -> int:
        return self._dec.get_drop_run()

    def is_gated(self) -> bool:
        return self._dec.is_gated()

    def set_strategy_kind(self, kind: str) -> None:
        with self._lock:
            self._enc.set_strategy_kind(kind)
            self._dec.set_strategy_kind(kind)

    def set_tune_offset(self, offset: int) -> None:
        with self._lock:
            self._dec.set_tune_offset(offset)

    def get_tune_offset(self) -> int:
        return self._dec.get_tune_offset()

    # ── public controls ──────────────────────────────────────────────────────
    def set_volume(self, vol: float) -> None:
        self._volume = float(vol)

    def set_source(self, source: str) -> None:
        self._source = source

    def get_payload_kind(self) -> str:
        return self._enc.get_payload_kind()

    def set_payload_kind(self, kind: str) -> None:
        with self._lock:
            self._enc.set_payload_kind(kind)
            self._dec.set_payload_kind(kind)

    def set_codec_mode(self, mode: SerializerMode) -> None:
        with self._lock:
            self._enc.set_codec_mode(mode)
            self._dec.set_codec_mode(mode)

    def set_sink_behaviour(self, behaviour: SinkBehaviour) -> None:
        with self._lock:
            self._dec.set_sink_behaviour(behaviour)

    def set_on_image(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_image(callback)

    def set_on_raw_image(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_raw_image(callback)

    def set_on_data(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_data(callback)

    def set_on_raw_data(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_raw_data(callback)

    def set_on_text(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_text(callback)

    def set_on_raw_text(self, callback: Optional[Callable]) -> None:
        with self._lock:
            self._dec.set_on_raw_text(callback)

    def get_latest_image(self):
        with self._lock:
            return self._dec.get_latest_image()

    def get_latest_bytes(self) -> Optional[bytes]:
        with self._lock:
            return self._dec.get_latest_bytes()

    def get_latest_text(self) -> Optional[str]:
        with self._lock:
            return self._dec.get_latest_text()

    def dump_decoded_audio_to_wav(self, file_path: str) -> None:
        with self._lock:
            self._dec.dump_decoded_audio_to_wav(file_path)

    def dump_decoded_bytes_to_file(self, file_path: str) -> None:
        with self._lock:
            self._dec.dump_decoded_bytes_to_file(file_path)

    def load_payload_file(self, file_path: str) -> None:
        with self._lock:
            self._enc.load_payload_file(file_path)

    def get_payload_path(self, kind: Optional[str] = None) -> Optional[str]:
        with self._lock:
            return self._enc.get_payload_path(kind)

    def get_position_fraction(self) -> float:
        with self._lock:
            return self._enc.get_position_fraction()

    def set_position_fraction(self, fraction: float) -> None:
        with self._lock:
            self._enc.set_position_fraction(fraction)

    def set_f0(self, f0: float) -> None:
        """Set encoder and decoder pitch together (initial tuning)."""
        self._enc.set_f0(f0)
        self._dec.set_f0(f0)

    def set_encoder_f0(self, f0: float) -> None:
        self._enc.set_f0(f0)

    def set_decoder_f0(self, f0: float) -> None:
        # Only takes effect in manual f0 mode; the estimator modes overwrite
        # the decoder's f0 per chunk from the received audio.
        self._dec.set_f0(f0)

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        with self._lock:
            self._enc.set_bits_per_symbol(bits_per_symbol)
            self._dec.set_bits_per_symbol(bits_per_symbol)

    def _callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            gate_active = self._midi_file_player.is_playing()
            note_held = True
            if gate_active:
                # Same gating contract as the split encoder GUI: pitch follows
                # the held note (both DSPs, so a manual-f0 decoder stays in
                # tune), the encoder keeps running through rests, and silence
                # is applied to the samples — so the decoder hears the same
                # gaps a real listener would.
                midi_note = self._note_state.current_note_or(-1)
                note_held = midi_note >= 0
                if note_held:
                    f0 = midi_note_to_hz(midi_note)
                    self._enc.set_f0(f0)
                    self._dec.set_f0(f0)

            enc_chunk = self._enc.process(frames)
            enc_samples = np.array(enc_chunk.get_samples(), dtype=np.float32)
            if gate_active and not note_held:
                enc_samples[:] = 0.0

            dec_samples = self._dec.process_chunk(enc_samples, frames)

            samples = enc_samples if self._source == "encoder" else dec_samples
            arr = np.array(samples, dtype=np.float32) * self._volume
            outdata[:, 0] = arr

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed.\nRun: pip install sounddevice")
        self._dec.reset()
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
    def __init__(self):
        super().__init__()
        self.title("SteganographySynthFlow")
        self.resizable(False, False)

        settings = Settings()
        settings.validate()
        self._settings = settings
        self._engine = AudioEngine(settings)
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
        # Guards against the estimated-f0 poll writing the slider, which
        # triggers _on_decoder_pitch_change and would feed the estimate back
        # into the engine (and clobber the user's manual pitch).
        self._syncing_pitch_ui = False
        # ttk.Scale.set() fires the slider's command, so the seeding .set()
        # calls in _build_ui would invoke handlers that reach for widgets not
        # built yet. The engine is already at its defaults at this point, so
        # those callbacks have nothing to do.
        self._ui_ready = False

        self._build_ui()
        self._ui_ready = True
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_position()
        self._poll_image()
        self._poll_decoded_text()
        self._poll_estimated_f0()
        self._poll_midi_playback()

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
        self._signal_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self._signal_var, foreground="#b00").pack(
            side="left", padx=12, pady=4
        )
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # ── start / stop ─────────────────────────────────────────────────────
        self._toggle_btn = ttk.Button(self, text="▶  Start", command=self._toggle, width=14)
        self._toggle_btn.grid(row=1, column=0, columnspan=2, **pad)

        # ── left column ─────────────────────────────────────────────────────
        left = ttk.Frame(self)
        left.grid(row=2, column=0, sticky="new")
        left.columnconfigure(0, weight=1)

        # ── payload kind ──────────────────────────────────────────────────────
        kind_frame = ttk.LabelFrame(left, text="Payload Type", padding=8)
        kind_frame.grid(row=0, column=0, sticky="ew", **pad)

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

        # ── payload file + position ──────────────────────────────────────────
        payload_frame = ttk.LabelFrame(left, text="Payload", padding=8)
        payload_frame.grid(row=1, column=0, sticky="ew", **pad)
        payload_frame.columnconfigure(0, weight=1)
        self._payload_frame = payload_frame

        self._payload_var = tk.StringVar(
            value=os.path.basename(self._settings.modulator_wav_path)
        )
        ttk.Label(payload_frame, textvariable=self._payload_var, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(payload_frame, text="Browse...", command=self._on_pick_payload).grid(
            row=0, column=1
        )

        self._payload_dragging = False
        self._position_slider = ttk.Scale(
            payload_frame, from_=0, to=self._settings.position_slider_max,
            orient="horizontal", length=self._settings.slider_length_px,
        )
        self._position_slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self._position_slider.bind("<ButtonPress-1>", self._on_position_drag_start)
        self._position_slider.bind("<ButtonRelease-1>", self._on_position_drag_end)

        # ── image codec mode (digital / analogue) ────────────────────────────
        codec_frame = ttk.LabelFrame(left, text="Image Encoding", padding=8)
        codec_frame.grid(row=2, column=0, sticky="ew", **pad)
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
        sink_frame = ttk.LabelFrame(left, text="Reconstruction Mode", padding=8)
        sink_frame.grid(row=3, column=0, sticky="ew", **pad)
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

        # ── output source ────────────────────────────────────────────────────
        src_frame = ttk.LabelFrame(left, text="Output Source", padding=8)
        src_frame.grid(row=4, column=0, sticky="ew", **pad)

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
        vol_frame = ttk.LabelFrame(left, text="Volume", padding=8)
        vol_frame.grid(row=5, column=0, sticky="ew", **pad)

        self._vol_label = ttk.Label(vol_frame, text="0 dB", width=6, anchor="e")
        self._vol_label.grid(row=0, column=1, padx=(6, 0))

        # Slider position is in dB, from volume_min_db to volume_max_db. Min = silence.
        self._vol_slider = ttk.Scale(
            vol_frame, from_=self._settings.volume_min_db, to=self._settings.volume_max_db,
            orient="horizontal", length=self._settings.slider_length_px,
            command=self._on_volume_change,
        )
        self._vol_slider.set(self._settings.volume_default_db)
        self._vol_slider.grid(row=0, column=0)

        # ── bits per symbol ───────────────────────────────────────────────────
        bits_frame = ttk.LabelFrame(left, text="Bits per Symbol", padding=8)
        bits_frame.grid(row=6, column=0, sticky="ew", **pad)

        self._bits_var = tk.StringVar(value=str(self._settings.bits_per_symbol))
        bits_combo = ttk.Combobox(
            bits_frame, textvariable=self._bits_var,
            values=[str(i) for i in range(self._settings.bits_per_symbol_min, self._settings.bits_per_symbol_max + 1)],
            state="readonly", width=6,
        )
        bits_combo.grid(row=0, column=0, padx=8)
        bits_combo.bind("<<ComboboxSelected>>", self._on_bits_change)

        # ── right column ────────────────────────────────────────────────────
        right = ttk.Frame(self)
        right.grid(row=2, column=1, sticky="new")
        right.columnconfigure(0, weight=1)

        # ── reconstructed image preview ──────────────────────────────────────
        preview_frame = ttk.LabelFrame(right, text="Reconstructed Image", padding=8)
        preview_frame.grid(row=0, column=0, sticky="ew", **pad)
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

        # ── f0 estimator ─────────────────────────────────────────────────────
        f0_frame = ttk.LabelFrame(right, text="F0 Estimator (decode)", padding=8)
        f0_frame.grid(row=2, column=0, sticky="ew", **pad)

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

        # ── encoder pitch ─────────────────────────────────────────────────────
        pitch_frame = ttk.LabelFrame(right, text="Encoder Pitch (Hz)", padding=8)
        pitch_frame.grid(row=3, column=0, sticky="ew", **pad)

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

        # ── decoder pitch + window tuning ─────────────────────────────────────
        # Split from the encoder pitch on purpose: detuning the decoder against
        # a known encoder pitch is how you measure the decoder's f0 tolerance.
        dec_tune_frame = ttk.LabelFrame(right, text="Decoder Pitch / Tuning", padding=8)
        dec_tune_frame.grid(row=4, column=0, sticky="ew", **pad)

        ttk.Label(dec_tune_frame, text="Pitch").grid(row=0, column=0, sticky="w")
        self._dec_pitch_label = ttk.Label(
            dec_tune_frame, text=f"{self._settings.pitch_default_hz:.0f} Hz", width=8, anchor="e",
        )
        self._dec_pitch_label.grid(row=0, column=2, padx=(6, 0))

        self._dec_pitch_slider = ttk.Scale(
            dec_tune_frame, from_=self._settings.pitch_min_hz, to=self._settings.pitch_max_hz,
            orient="horizontal", length=self._settings.slider_length_px,
            command=self._on_decoder_pitch_change,
        )
        self._dec_pitch_slider.set(self._settings.pitch_default_hz)
        self._dec_pitch_slider.grid(row=0, column=1)

        self._link_pitch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            dec_tune_frame, text="Follow encoder pitch", variable=self._link_pitch_var,
            command=self._on_link_pitch_change,
        ).grid(row=1, column=1, sticky="w", pady=(2, 6))

        ttk.Label(dec_tune_frame, text="Offset").grid(row=2, column=0, sticky="w")
        self._tune_label = ttk.Label(dec_tune_frame, text="0", width=8, anchor="e")
        self._tune_label.grid(row=2, column=2, padx=(6, 0))

        # Slides the decode window within a chunk, so 0..chunk_size-1 covers
        # every alignment; the range is restated when the strategy changes.
        self._tune_slider = ttk.Scale(
            dec_tune_frame, from_=0, to=self._settings.chunk_size - 1,
            orient="horizontal", length=self._settings.slider_length_px,
            command=self._on_tune_change,
        )
        self._tune_slider.set(0)
        self._tune_slider.grid(row=2, column=1)

        # ── midi file playback ───────────────────────────────────────────────
        midi_file_frame = ttk.LabelFrame(right, text="MIDI File Playback", padding=8)
        midi_file_frame.grid(row=5, column=0, sticky="ew", **pad)
        midi_file_frame.columnconfigure(0, weight=1)

        self._midi_file_var = tk.StringVar(value="(no file)")
        ttk.Label(midi_file_frame, textvariable=self._midi_file_var, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(midi_file_frame, text="Browse...", command=self._on_pick_midi_file).grid(
            row=0, column=1
        )

        self._midi_play_btn = ttk.Button(
            midi_file_frame, text="▶  Play", command=self._on_midi_play_toggle, width=10,
        )
        self._midi_play_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))

        self._midi_loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            midi_file_frame, text="Loop", variable=self._midi_loop_var,
            command=self._on_midi_loop_change,
        ).grid(row=1, column=1, sticky="w", pady=(6, 0))

        transpose_row = ttk.Frame(midi_file_frame)
        transpose_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(transpose_row, text="Transpose (semitones)").grid(row=0, column=0, padx=(0, 8))
        self._transpose_var = tk.StringVar(value="0")
        transpose_spin = ttk.Spinbox(
            transpose_row, textvariable=self._transpose_var,
            from_=self._settings.midi_transpose_min, to=self._settings.midi_transpose_max,
            increment=1, width=5, command=self._on_transpose_change,
        )
        transpose_spin.grid(row=0, column=1)
        transpose_spin.bind("<Return>", self._on_transpose_change)
        transpose_spin.bind("<FocusOut>", self._on_transpose_change)

        ttk.Label(midi_file_frame, text="Tempo").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self._tempo_label = ttk.Label(
            midi_file_frame, text=f"×{self._settings.midi_tempo_scale_default:.2f}", width=6, anchor="e",
        )
        self._tempo_label.grid(row=4, column=1, padx=(6, 0))

        self._tempo_slider = ttk.Scale(
            midi_file_frame,
            from_=self._settings.midi_tempo_scale_min, to=self._settings.midi_tempo_scale_max,
            orient="horizontal", length=self._settings.slider_length_px,
            command=self._on_tempo_change,
        )
        self._tempo_slider.set(self._settings.midi_tempo_scale_default)
        self._tempo_slider.grid(row=4, column=0, sticky="w")

        self._piano = PianoKeyboard(
            midi_file_frame,
            low_note=self._settings.piano_low_note, high_note=self._settings.piano_high_note,
        )
        self._piano.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self._update_kind_dependent_visibility()
        self._update_decoder_pitch_state()

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
        kind = self._kind_var.get()
        if kind == "image":
            file_path = filedialog.askopenfilename(
                title="Select image payload",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                    ("All files", "*.*"),
                ],
            )
        elif kind == "text":
            file_path = filedialog.askopenfilename(
                title="Select text payload",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
        elif kind == "binary":
            file_path = filedialog.askopenfilename(
                title="Select binary payload",
                filetypes=[("All files", "*.*")],
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
        fraction = self._position_slider.get() / self._settings.position_slider_max
        self._engine.set_position_fraction(fraction)
        self._payload_dragging = False

    def _poll_position(self) -> None:
        if not self._payload_dragging:
            fraction = self._engine.get_position_fraction()
            self._position_slider.set(fraction * self._settings.position_slider_max)
        self.after(self._settings.gui_poll_interval_ms, self._poll_position)

    def _on_strategy_change(self) -> None:
        self._engine.set_strategy_kind(self._strategy_var.get())
        # chunk_size follows the strategy, and the tune offset is taken mod
        # chunk_size, so the slider's range has to follow it too.
        self._tune_slider.configure(to=self._settings.chunk_size - 1)
        self._tune_slider.set(self._engine.get_tune_offset())
        self._tune_label.configure(text=str(self._engine.get_tune_offset()))

    def _on_kind_change(self) -> None:
        kind = self._kind_var.get()
        self._engine.set_payload_kind(kind)
        default_name = os.path.basename(self._engine.get_payload_path(kind) or "")
        self._payload_var.set(default_name)
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
        # Called from the audio callback thread; hand off to the Tk main loop.
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
            self._pending_image_frame = None
            photo = self._render_frame(frame, self._preview_label)
            if photo is not None:
                self._preview_photo = photo
        raw_frame = self._pending_raw_frame
        if raw_frame is not None:
            self._pending_raw_frame = None
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

    def _on_source_change(self) -> None:
        self._engine.set_source(self._source_var.get())

    def _on_volume_change(self, value: str) -> None:
        db = float(value)
        self._settings.volume_default_db = db
        gain = 0.0 if db <= self._settings.volume_min_db else 10 ** (db / 20.0)
        self._engine.set_volume(gain)
        label = "−∞ dB" if db <= self._settings.volume_min_db else f"{db:.0f} dB"
        self._vol_label.configure(text=label)

    def _on_pitch_change(self, value: str) -> None:
        if not self._ui_ready:
            return
        f0 = float(value)
        self._settings.pitch_default_hz = f0
        self._engine.set_encoder_f0(f0)
        self._pitch_label.configure(text=f"{f0:.2f} Hz")
        if self._link_pitch_var.get():
            self._set_decoder_pitch_ui(f0)
            self._engine.set_decoder_f0(f0)

    def _on_decoder_pitch_change(self, value: str) -> None:
        if self._syncing_pitch_ui or not self._ui_ready:
            return
        f0 = float(value)
        self._engine.set_decoder_f0(f0)
        self._dec_pitch_label.configure(text=f"{f0:.2f} Hz")

    def _set_decoder_pitch_ui(self, f0: float) -> None:
        """Move the decoder pitch slider without it feeding back into the engine."""
        self._syncing_pitch_ui = True
        try:
            self._dec_pitch_slider.set(
                min(max(f0, self._settings.pitch_min_hz), self._settings.pitch_max_hz)
            )
        finally:
            self._syncing_pitch_ui = False
        self._dec_pitch_label.configure(text=f"{f0:.2f} Hz")

    def _on_link_pitch_change(self) -> None:
        self._update_decoder_pitch_state()
        if self._link_pitch_var.get():
            f0 = float(self._pitch_slider.get())
            self._set_decoder_pitch_ui(f0)
            self._engine.set_decoder_f0(f0)

    def _update_decoder_pitch_state(self) -> None:
        # The decoder pitch slider is only meaningful in manual f0 mode and
        # when not slaved to the encoder; the estimator modes drive it instead.
        manual = self._f0_mode_var.get() == "Manual"
        editable = manual and not self._link_pitch_var.get()
        self._dec_pitch_slider.configure(state="normal" if editable else "disabled")

    def _on_tune_change(self, value: str) -> None:
        if not self._ui_ready:
            return
        offset = int(float(value))
        self._engine.set_tune_offset(offset)
        self._tune_label.configure(text=str(self._engine.get_tune_offset()))

    _F0_MODE_TO_KEY = {"Manual": "manual", "Autocorrelation": "autocorr", "FFT": "fft"}

    def _on_f0_mode_change(self, _event=None) -> None:
        mode = self._F0_MODE_TO_KEY[self._f0_mode_var.get()]
        self._engine.set_f0_estimator_mode(mode)
        self._update_decoder_pitch_state()

    def _on_quantize_change(self) -> None:
        self._engine.set_pitch_quantize(self._quantize_var.get())

    def _poll_estimated_f0(self) -> None:
        is_manual = self._f0_mode_var.get() == "Manual"
        if not is_manual:
            # The estimate is the decoder's own pitch, so it drives the decoder
            # slider; the encoder slider stays where the user put it.
            f0 = self._engine.get_estimated_f0()
            if f0 > 0.0:
                self._set_decoder_pitch_ui(f0)

        drop_run = self._engine.get_drop_run()
        if self._engine.is_gated() or drop_run > 0:
            self._signal_var.set(f"SIGNAL WEAK — drop {drop_run}/{self._settings.drop_tolerance_chunks}")
        elif not is_manual:
            self._signal_var.set(f"confidence {self._engine.get_confidence():.2f}")
        else:
            self._signal_var.set("")
        self.after(self._settings.gui_poll_interval_ms, self._poll_estimated_f0)

    def _on_pick_midi_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select MIDI file",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            self._engine.get_midi_file_player().load(file_path)
        except Exception as exc:
            messagebox.showerror("MIDI File Error", str(exc))
            return
        self._midi_file_var.set(os.path.basename(file_path))

    def _on_midi_play_toggle(self) -> None:
        player = self._engine.get_midi_file_player()
        if player.is_playing():
            player.stop()
        else:
            try:
                player.start()
            except Exception as exc:
                messagebox.showerror("MIDI File Error", str(exc))
                return
        self._update_midi_playback_ui()

    def _on_tempo_change(self, value: str) -> None:
        if not self._ui_ready:
            return
        scale = float(value)
        self._engine.get_midi_file_player().set_tempo_scale(scale)
        self._tempo_label.configure(text=f"×{scale:.2f}")

    def _on_midi_loop_change(self) -> None:
        self._engine.get_midi_file_player().set_loop(self._midi_loop_var.get())

    def _on_transpose_change(self, _event=None) -> None:
        player = self._engine.get_midi_file_player()
        try:
            semitones = int(self._transpose_var.get())
        except ValueError:
            self._transpose_var.set(str(player.get_transpose()))
            return
        semitones = min(max(semitones, self._settings.midi_transpose_min),
                        self._settings.midi_transpose_max)
        self._transpose_var.set(str(semitones))
        player.set_transpose(semitones)

    def _update_midi_playback_ui(self) -> None:
        playing = self._engine.get_midi_file_player().is_playing()
        self._midi_play_btn.configure(text="⏹  Stop" if playing else "▶  Play")
        # During playback the notes drive the encoder pitch, so the manual
        # slider is taken out of play rather than fighting the file.
        self._pitch_slider.configure(state="disabled" if playing else "normal")

    def _poll_midi_playback(self) -> None:
        # Playback ends on its own when the file runs out, so the Play button
        # and pitch-slider state have to be re-derived on every poll.
        self._update_midi_playback_ui()
        self._piano.set_active_note(self._engine.get_active_note())
        if self._engine.get_midi_file_player().is_playing():
            note = self._engine.get_active_note()
            if note is not None:
                self._pitch_label.configure(text=f"{midi_note_to_hz(note):.2f} Hz")
        self.after(self._settings.gui_note_poll_interval_ms, self._poll_midi_playback)

    def _on_close(self) -> None:
        self._engine.get_midi_file_player().stop()
        self._stop()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
