"""Bottom keyboard bar: playable piano plus every note source that can drive
the encoder pitch (mouse, QWERTY, MIDI device, MIDI file playback)."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from gui.theme import Palette
from gui.widgets import LabeledScale, SynthPiano
from KeyboardNoteInput import KeyboardNoteInput
from NoteState import midi_note_to_hz
from Settings import Settings


class KeyboardBar(ttk.Frame):
    """Spans the bottom of the rack whenever an encoder module is present.

    Works against any engine exposing the encoder-side note API
    (EncoderEngine or LinkedEngine).
    """

    def __init__(self, parent, engine, settings: Settings, *,
                 on_note_pitch: Optional[Callable[[bool, float], None]] = None):
        super().__init__(parent, padding=(12, 8))
        self._engine = engine
        self._settings = settings
        self._on_note_pitch = on_note_pitch  # (gate_active, current_f0)
        self._alive = True
        self._keyboard_input = KeyboardNoteInput(self.winfo_toplevel(),
                                                 engine.get_note_state())

        self.columnconfigure(0, weight=1)

        # ── control strip above the keys ────────────────────────────────────
        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # QWERTY
        self._qwerty_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="QWERTY keys", variable=self._qwerty_var,
                        command=self._on_qwerty_toggle).pack(side="left")

        ttk.Label(controls, text="│", style="Dim.TLabel").pack(side="left", padx=10)

        # MIDI device
        self._midi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="MIDI in", variable=self._midi_var,
                        command=self._on_midi_toggle).pack(side="left")
        midi_names = ["(default)"] + engine.list_midi_devices()
        self._midi_device_var = tk.StringVar(value=midi_names[0])
        self._midi_combo = ttk.Combobox(controls, textvariable=self._midi_device_var,
                                        values=midi_names, state="readonly", width=22)
        self._midi_combo.pack(side="left", padx=(6, 0))

        ttk.Label(controls, text="│", style="Dim.TLabel").pack(side="left", padx=10)

        # MIDI file transport
        self._midi_file_var = tk.StringVar(value="(no file)")
        file_well = tk.Label(controls, textvariable=self._midi_file_var, anchor="w",
                             bg=Palette.INSET, fg=Palette.TEXT,
                             font=Palette.FONT_MONO, padx=6, pady=2, width=18)
        file_well.pack(side="left")
        ttk.Button(controls, text="Load MIDI…",
                   command=self._on_pick_midi_file).pack(side="left", padx=(6, 6))
        self._play_btn = ttk.Button(controls, text="▶ Play", width=8,
                                    command=self._on_play_toggle)
        self._play_btn.pack(side="left")
        self._loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Loop", variable=self._loop_var,
                        command=self._on_loop).pack(side="left", padx=(8, 0))

        ttk.Label(controls, text="Transpose", style="Dim.TLabel").pack(
            side="left", padx=(14, 4))
        self._transpose_var = tk.StringVar(value="0")
        transpose = ttk.Spinbox(controls, textvariable=self._transpose_var,
                                from_=settings.midi_transpose_min,
                                to=settings.midi_transpose_max,
                                increment=1, width=4, command=self._on_transpose)
        transpose.pack(side="left")
        transpose.bind("<Return>", self._on_transpose)
        transpose.bind("<FocusOut>", self._on_transpose)

        self._tempo = LabeledScale(
            controls, "Tempo", settings.midi_tempo_scale_min,
            settings.midi_tempo_scale_max, fmt=lambda v: f"×{v:.2f}",
            command=self._on_tempo, init=settings.midi_tempo_scale_default,
            length=110)
        self._tempo.pack(side="left", padx=(14, 0))

        # ── the keyboard itself ─────────────────────────────────────────────
        piano_holder = ttk.Frame(self)
        piano_holder.grid(row=1, column=0)
        self._piano = SynthPiano(
            piano_holder,
            low_note=settings.piano_low_note, high_note=settings.piano_high_note,
            on_note_on=self._on_piano_press, on_note_off=self._on_piano_release,
        )
        self._piano.pack()

        self._poll_notes()

    # ── mouse piano ────────────────────────────────────────────────────────
    def _on_piano_press(self, note: int) -> None:
        self._engine.set_pointer_active(True)
        self._engine.get_note_state().note_on(note)

    def _on_piano_release(self, note: int) -> None:
        self._engine.get_note_state().note_off(note)
        self._engine.set_pointer_active(False)

    # ── QWERTY / MIDI device ───────────────────────────────────────────────
    def _on_qwerty_toggle(self) -> None:
        enabled = self._qwerty_var.get()
        if enabled:
            self._keyboard_input.enable()
        else:
            self._keyboard_input.disable()
        self._engine.set_keyboard_enabled(enabled)

    def _selected_midi_device(self) -> Optional[str]:
        name = self._midi_device_var.get()
        return name if name != "(default)" else None

    def _on_midi_toggle(self) -> None:
        enabled = self._midi_var.get()
        try:
            self._engine.set_midi_enabled(enabled, self._selected_midi_device())
        except Exception as exc:
            messagebox.showerror("MIDI Error", str(exc))
            self._midi_var.set(False)
            return
        self._midi_combo.configure(state="disabled" if enabled else "readonly")

    # ── MIDI file playback ─────────────────────────────────────────────────
    def _on_pick_midi_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select MIDI file",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            self._engine.get_midi_file_player().load(file_path)
        except Exception as exc:
            messagebox.showerror("MIDI File Error", str(exc))
            return
        self._midi_file_var.set(os.path.basename(file_path))

    def _on_play_toggle(self) -> None:
        player = self._engine.get_midi_file_player()
        if player.is_playing():
            player.stop()
        else:
            try:
                player.start()
            except Exception as exc:
                messagebox.showerror("MIDI File Error", str(exc))
                return

    def _on_tempo(self, scale: float) -> None:
        self._engine.get_midi_file_player().set_tempo_scale(scale)

    def _on_loop(self) -> None:
        self._engine.get_midi_file_player().set_loop(self._loop_var.get())

    def _on_transpose(self, _event=None) -> None:
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

    # ── polling ────────────────────────────────────────────────────────────
    def _poll_notes(self) -> None:
        if not self._alive:
            return
        playing = self._engine.get_midi_file_player().is_playing()
        self._play_btn.configure(text="⏹ Stop" if playing else "▶ Play")

        note = self._engine.get_active_note()
        self._piano.set_active_note(note)

        gate_active = (self._qwerty_var.get() or self._midi_var.get() or playing
                       or note is not None)
        if self._on_note_pitch is not None:
            f0 = (midi_note_to_hz(note) if note is not None
                  else self._engine.get_encoder_f0())
            self._on_note_pitch(gate_active, f0)
        self.after(self._settings.gui_note_poll_interval_ms, self._poll_notes)

    def shutdown(self) -> None:
        self._keyboard_input.disable()

    def destroy(self) -> None:
        self._alive = False
        self._keyboard_input.disable()
        super().destroy()
