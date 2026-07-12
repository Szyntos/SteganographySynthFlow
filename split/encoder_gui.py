import os
import sys
import threading
import time
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

from audio_callback_diag import log_callback_event
from EncoderDSP import EncoderDSP
from KeyboardNoteInput import KeyboardNoteInput
from MidiDeviceInput import MidiDeviceInput, list_midi_input_devices
from NoteState import NoteState, midi_note_to_hz
from PianoKeyboard import PianoKeyboard
from SerializerMode import SerializerMode
from Settings import Settings


def list_output_devices() -> List[Tuple[int, str]]:
    if sd is None:
        return []
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            devices.append((idx, f"[{idx}] {dev['name']}"))
    return devices


class EncoderEngine:
    """Drives an EncoderDSP against a live audio output device stream, with
    MIDI/keyboard note gating on top.

    All strategy/payload/codec assembly lives in EncoderDSP; this class only
    owns the sounddevice stream, device selection, note-input gating, and
    diagnostics.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._volume: float = 1.0
        self._stream = None
        self._lock = threading.Lock()
        self._output_device: Optional[int] = None

        self._dsp = EncoderDSP(settings)
        self._dsp.set_f0(settings.pitch_default_hz)

        self._note_state = NoteState()
        self._midi_input = MidiDeviceInput(self._note_state)
        self._midi_enabled: bool = False
        self._keyboard_enabled: bool = False

    # ── public controls ──────────────────────────────────────────────────────
    def set_volume(self, vol: float) -> None:
        self._volume = float(vol)

    def set_output_device(self, device: Optional[int]) -> None:
        self._output_device = device

    def is_running(self) -> bool:
        return self._stream is not None

    def set_strategy_kind(self, kind: str) -> None:
        with self._lock:
            self._dsp.set_strategy_kind(kind)

    def set_payload_kind(self, kind: str) -> None:
        with self._lock:
            self._dsp.set_payload_kind(kind)

    def set_codec_mode(self, mode: SerializerMode) -> None:
        with self._lock:
            self._dsp.set_codec_mode(mode)

    def load_payload_file(self, file_path: str) -> None:
        with self._lock:
            self._dsp.load_payload_file(file_path)

    def get_payload_path(self, kind: Optional[str] = None) -> Optional[str]:
        with self._lock:
            return self._dsp.get_payload_path(kind)

    def get_position_fraction(self) -> float:
        with self._lock:
            return self._dsp.get_position_fraction()

    def set_position_fraction(self, fraction: float) -> None:
        with self._lock:
            self._dsp.set_position_fraction(fraction)

    def set_f0(self, f0: float) -> None:
        self._dsp.set_f0(f0)

    def get_f0(self) -> float:
        return self._dsp.get_f0()

    def list_midi_devices(self) -> List[str]:
        return list_midi_input_devices()

    def set_midi_enabled(self, enabled: bool, device_name: Optional[str] = None) -> None:
        with self._lock:
            if enabled:
                self._midi_input.start(device_name)
            else:
                self._midi_input.stop()
            self._midi_enabled = enabled

    def is_midi_enabled(self) -> bool:
        return self._midi_enabled

    def get_note_state(self) -> NoteState:
        return self._note_state

    def set_keyboard_enabled(self, enabled: bool) -> None:
        self._keyboard_enabled = bool(enabled)

    def get_active_note(self) -> Optional[int]:
        note = self._note_state.current_note_or(-1)
        return note if note >= 0 else None

    def set_bits_per_symbol(self, bits_per_symbol: int) -> None:
        with self._lock:
            self._dsp.set_bits_per_symbol(bits_per_symbol)

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        _cb_start = time.perf_counter()
        try:
            with self._lock:
                gate_active = self._midi_enabled or self._keyboard_enabled
                note_held = True
                if gate_active:
                    # Read the held-note stack once per audio block (chunk
                    # boundary) — set_f0 only updates the scalar frequency used
                    # by the next block; it never resets the running harmonic
                    # phases, so pitch changes stay click-free. If no note is
                    # held, keep the last f0 instead of jumping to a fallback:
                    # the encoder must keep advancing its carrier phase and
                    # payload position uninterrupted (no discontinuity when a
                    # note resumes), so silence is applied only to the output
                    # samples below, never by pausing the encoder itself.
                    midi_note = self._note_state.current_note_or(-1)
                    note_held = midi_note >= 0
                    if note_held:
                        self._dsp.set_f0(midi_note_to_hz(midi_note))

                enc_chunk = self._dsp.process(frames)
                arr = np.array(enc_chunk.get_samples(), dtype=np.float32) * self._volume
                if gate_active and not note_held:
                    arr[:] = 0.0
                outdata[:, 0] = arr
        finally:
            duration = time.perf_counter() - _cb_start
            budget = frames / float(self._settings.fs_out)
            log_callback_event("encoder", status, duration, budget)

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

    def shutdown_midi(self) -> None:
        self._midi_input.stop()
        self._midi_enabled = False


class EncoderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SSF Encoder")
        self.resizable(False, False)

        settings = Settings()
        settings.validate()
        self._settings = settings
        self._engine = EncoderEngine(settings)
        self._running = False
        self._devices = list_output_devices()
        self._keyboard_input = KeyboardNoteInput(self, self._engine.get_note_state())

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_position()
        self._poll_note_control()

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 6}

        self._status_var = tk.StringVar(value="Stopped")
        status_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 0))
        ttk.Label(status_frame, text="Status:").pack(side="left", padx=6, pady=4)
        ttk.Label(status_frame, textvariable=self._status_var, font=("", 10, "bold")).pack(
            side="left", pady=4
        )
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self._toggle_btn = ttk.Button(self, text="▶  Start", command=self._toggle, width=14)
        self._toggle_btn.grid(row=1, column=0, columnspan=2, **pad)

        # ── left column ─────────────────────────────────────────────────────
        left = ttk.Frame(self)
        left.grid(row=2, column=0, sticky="new")
        left.columnconfigure(0, weight=1)

        # ── output device ────────────────────────────────────────────────────
        dev_frame = ttk.LabelFrame(left, text="Output Device", padding=8)
        dev_frame.grid(row=0, column=0, sticky="ew", **pad)
        dev_frame.columnconfigure(0, weight=1)

        names = ["(default)"] + [name for _, name in self._devices]
        self._device_var = tk.StringVar(value=names[0])
        self._device_combo = ttk.Combobox(
            dev_frame, textvariable=self._device_var, values=names, state="readonly", width=42
        )
        self._device_combo.grid(row=0, column=0, sticky="ew")
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_change)

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

        # ── payload file + position ──────────────────────────────────────────
        payload_frame = ttk.LabelFrame(left, text="Payload", padding=8)
        payload_frame.grid(row=2, column=0, sticky="ew", **pad)
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
            payload_frame, from_=0, to=self._settings.position_slider_max,
            orient="horizontal", length=self._settings.slider_length_px,
        )
        self._position_slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self._position_slider.bind("<ButtonPress-1>", self._on_position_drag_start)
        self._position_slider.bind("<ButtonRelease-1>", self._on_position_drag_end)

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

        # ── volume ───────────────────────────────────────────────────────────
        vol_frame = ttk.LabelFrame(left, text="Volume", padding=8)
        vol_frame.grid(row=4, column=0, sticky="ew", **pad)

        self._vol_label = ttk.Label(vol_frame, text="0 dB", width=6, anchor="e")
        self._vol_label.grid(row=0, column=1, padx=(6, 0))

        self._vol_slider = ttk.Scale(
            vol_frame, from_=self._settings.volume_min_db, to=self._settings.volume_max_db,
            orient="horizontal", length=self._settings.slider_length_px,
            command=self._on_volume_change,
        )
        self._vol_slider.set(self._settings.volume_default_db)
        self._vol_slider.grid(row=0, column=0)

        # ── bits per symbol ───────────────────────────────────────────────────
        bits_frame = ttk.LabelFrame(left, text="Bits per Symbol", padding=8)
        bits_frame.grid(row=5, column=0, sticky="ew", **pad)

        self._bits_var = tk.StringVar(value=str(self._engine._settings.bits_per_symbol))
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

        # ── midi control ─────────────────────────────────────────────────────
        midi_frame = ttk.LabelFrame(right, text="MIDI Control (mono, last-note priority)", padding=8)
        midi_frame.grid(row=0, column=0, sticky="ew", **pad)
        midi_frame.columnconfigure(1, weight=1)

        self._midi_devices = self._engine.list_midi_devices()
        midi_names = ["(none)"] + self._midi_devices
        self._midi_device_var = tk.StringVar(value=midi_names[0])
        self._midi_device_combo = ttk.Combobox(
            midi_frame, textvariable=self._midi_device_var, values=midi_names,
            state="readonly", width=28,
        )
        self._midi_device_combo.grid(row=0, column=1, sticky="ew")

        self._midi_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            midi_frame, text="Enabled", variable=self._midi_enabled_var,
            command=self._on_midi_toggle,
        ).grid(row=0, column=0, padx=(0, 8))

        # ── keyboard control + on-screen piano ───────────────────────────────
        kbd_frame = ttk.LabelFrame(right, text="Keyboard Control (QWERTY piano)", padding=8)
        kbd_frame.grid(row=1, column=0, sticky="ew", **pad)

        self._keyboard_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            kbd_frame, text="Enabled  (Z-M / , . ; / and Q-P play notes, C4-E6)",
            variable=self._keyboard_enabled_var, command=self._on_keyboard_toggle,
        ).grid(row=0, column=0, sticky="w")

        self._piano = PianoKeyboard(
            kbd_frame, low_note=self._settings.piano_low_note, high_note=self._settings.piano_high_note,
        )
        self._piano.grid(row=1, column=0, sticky="w", pady=(8, 0))

        # ── pitch ─────────────────────────────────────────────────────────────
        pitch_frame = ttk.LabelFrame(right, text="Pitch (Hz)", padding=8)
        pitch_frame.grid(row=2, column=0, sticky="ew", **pad)

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

        self._update_kind_dependent_visibility()
        self._update_note_control_visibility()

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

    def _on_kind_change(self) -> None:
        kind = self._kind_var.get()
        self._engine.set_payload_kind(kind)
        default_name = os.path.basename(self._engine.get_payload_path(kind) or "")
        self._payload_var.set(default_name)
        self._update_kind_dependent_visibility()

    def _update_kind_dependent_visibility(self) -> None:
        state = "normal" if self._kind_var.get() != "audio" else "disabled"
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

    def _selected_midi_device(self) -> Optional[str]:
        name = self._midi_device_var.get()
        return name if name != "(none)" else None

    def _on_midi_toggle(self) -> None:
        enabled = self._midi_enabled_var.get()
        try:
            self._engine.set_midi_enabled(enabled, self._selected_midi_device())
        except Exception as exc:
            messagebox.showerror("MIDI Error", str(exc))
            self._midi_enabled_var.set(False)
            return
        self._update_note_control_visibility()

    def _on_keyboard_toggle(self) -> None:
        enabled = self._keyboard_enabled_var.get()
        if enabled:
            self._keyboard_input.enable()
        else:
            self._keyboard_input.disable()
        self._engine.set_keyboard_enabled(enabled)
        self._update_note_control_visibility()

    def _update_note_control_visibility(self) -> None:
        note_control_active = self._midi_enabled_var.get() or self._keyboard_enabled_var.get()
        self._midi_device_combo.configure(state="disabled" if self._midi_enabled_var.get() else "readonly")
        self._pitch_slider.configure(state="disabled" if note_control_active else "normal")

    def _poll_note_control(self) -> None:
        if self._midi_enabled_var.get() or self._keyboard_enabled_var.get():
            f0 = self._engine.get_f0()
            self._pitch_slider.set(min(max(f0, self._settings.pitch_min_hz), self._settings.pitch_max_hz))
            self._pitch_label.configure(text=f"{f0:.2f} Hz")
        self._piano.set_active_note(self._engine.get_active_note())
        self.after(self._settings.gui_note_poll_interval_ms, self._poll_note_control)

    def _on_close(self) -> None:
        self._keyboard_input.disable()
        self._engine.shutdown_midi()
        self._stop()
        self.destroy()


if __name__ == "__main__":
    app = EncoderApp()
    app.mainloop()
