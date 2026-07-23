"""Encoder rack module: transmission parameters, payload transport, pitch and
the synth voice (amp/filter envelopes + resonant low-pass)."""

import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from gui.settings_pane import DspSettingsPane
from gui.wave_editor import WaveEditorWindow, WaveViewer
from gui.widgets import (BipolarBar, FileRow, LabeledScale, Panel, Section,
                         Segmented, VBarBank)
from SerializerMode import SerializerMode
from Settings import Settings

_PAYLOAD_DIALOGS = {
    "audio": ("Select audio payload", [("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                                       ("All files", "*.*")]),
    "image": ("Select image payload", [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                                       ("All files", "*.*")]),
    "text": ("Select text payload", [("Text files", "*.txt"), ("All files", "*.*")]),
    "binary": ("Select binary payload", [("All files", "*.*")]),
}


class EncoderPanel(Panel):
    """UI for the encoder side. Works against EncoderEngine or LinkedEngine
    (both expose the same encoder-side API)."""

    def __init__(self, parent, engine, settings: Settings, *,
                 linked: bool,
                 output_devices,
                 on_close: Callable[[], None],
                 on_strategy_change: Optional[Callable[[str], None]] = None,
                 on_kind_change: Optional[Callable[[str], None]] = None,
                 on_device_change: Optional[Callable[[], None]] = None,
                 on_pitch_change: Optional[Callable[[float], None]] = None):
        super().__init__(parent, "ENCODER", on_close=on_close)
        self._engine = engine
        self._settings = settings
        self._linked = linked
        self._on_strategy_change = on_strategy_change
        self._on_kind_change = on_kind_change
        self._on_device_change = on_device_change
        self._on_pitch_change = on_pitch_change
        self._devices = output_devices
        self._payload_dragging = False
        self._alive = True

        body = self.body
        body.columnconfigure(0, weight=1)

        # ── output device ───────────────────────────────────────────────────
        dev = Section(body, "Output Device")
        dev.grid(row=0, column=0, sticky="ew")
        dev.content.columnconfigure(0, weight=1)
        names = ["(default)"] + [name for _, name in self._devices]
        self._device_var = tk.StringVar(value=names[0])
        combo = ttk.Combobox(dev.content, textvariable=self._device_var,
                             values=names, state="readonly")
        combo.grid(row=0, column=0, sticky="ew")
        combo.bind("<<ComboboxSelected>>", self._on_device_selected)

        # ── 2nd output (standalone only; linked shares one stream) ─────────
        if not linked:
            sec = ttk.Frame(dev.content)
            sec.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            sec.columnconfigure(1, weight=1)
            self._out2_enabled_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(sec, text="2nd output", variable=self._out2_enabled_var,
                            command=self._on_output2_toggle).grid(
                row=0, column=0, sticky="w")
            default2 = self._find_cable_device_name(names) or (
                names[1] if len(names) > 1 else names[0])
            self._device2_var = tk.StringVar(value=default2)
            self._combo2 = ttk.Combobox(sec, textvariable=self._device2_var,
                                        values=names, state="disabled")
            self._combo2.grid(row=0, column=1, sticky="ew", padx=(6, 0))
            self._combo2.bind("<<ComboboxSelected>>", self._on_device2_selected)

        # ── transmission parameters ─────────────────────────────────────────
        tx = Section(body, "Transmission" + ("  (shared with decoder)" if linked else ""))
        tx.grid(row=1, column=0, sticky="ew")
        tx.content.columnconfigure(1, weight=1)

        ttk.Label(tx.content, text="Payload", style="Dim.TLabel", width=9).grid(
            row=0, column=0, sticky="w", pady=2)
        self._kind_seg = Segmented(
            tx.content,
            [("Audio", "audio"), ("Image", "image"), ("Binary", "binary"), ("Text", "text")],
            self._on_kind, "audio")
        self._kind_seg.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(tx.content, text="Split", style="Dim.TLabel", width=9).grid(
            row=1, column=0, sticky="w", pady=2)
        self._strategy_seg = Segmented(
            tx.content, [("Two-Split", "two"), ("Four-Split", "four")],
            self._on_strategy, "two")
        self._strategy_seg.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(tx.content, text="Codec", style="Dim.TLabel", width=9).grid(
            row=2, column=0, sticky="w", pady=2)
        self._codec_seg = Segmented(
            tx.content, [("Digital", "digital"), ("Analogue", "analogue")],
            self._on_codec, "analogue")
        self._codec_seg.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(tx.content, text="Bits/sym", style="Dim.TLabel", width=9).grid(
            row=3, column=0, sticky="w", pady=2)
        self._bits_var = tk.StringVar(value=str(settings.bits_per_symbol))
        bits_combo = ttk.Combobox(
            tx.content, textvariable=self._bits_var, state="readonly", width=5,
            values=[str(i) for i in range(settings.bits_per_symbol_min,
                                          settings.bits_per_symbol_max + 1)])
        bits_combo.grid(row=3, column=1, sticky="w", pady=2)
        bits_combo.bind("<<ComboboxSelected>>", self._on_bits)

        ttk.Label(tx.content, text="Smp/sym", style="Dim.TLabel", width=9).grid(
            row=4, column=0, sticky="w", pady=2)
        self._samples_var = tk.StringVar(value=str(settings.audio_samples_per_symbol))
        self._samples_combo = ttk.Combobox(
            tx.content, textvariable=self._samples_var, state="readonly", width=5,
            values=[str(2 ** i) for i in range(settings.bits_per_symbol_max.bit_length())])
        self._samples_combo.grid(row=4, column=1, sticky="w", pady=2)
        self._samples_combo.bind("<<ComboboxSelected>>", self._on_samples_per_symbol)

        # ── payload transport ───────────────────────────────────────────────
        transport = Section(body, "Payload File")
        transport.grid(row=2, column=0, sticky="ew")
        transport.content.columnconfigure(0, weight=1)

        self._file_row = FileRow(
            transport.content,
            os.path.basename(settings.modulator_wav_path),
            self._on_pick_payload)
        self._file_row.grid(row=0, column=0, sticky="ew")

        self._position = LabeledScale(
            transport.content, "Position", 0, settings.position_slider_max,
            fmt=lambda v: f"{v / settings.position_slider_max * 100:.0f}%",
            init=0)
        self._position.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        # Only pushed to the engine on release; while playing the engine drives it.
        self._position._scale.bind("<ButtonPress-1>", self._on_position_press)
        self._position._scale.bind("<ButtonRelease-1>", self._on_position_release)

        # ── pitch ───────────────────────────────────────────────────────────
        pitch = Section(body, "Pitch")
        pitch.grid(row=3, column=0, sticky="ew")
        pitch.content.columnconfigure(0, weight=1)
        self._pitch = LabeledScale(
            pitch.content, "f0", settings.pitch_min_hz, settings.pitch_max_hz,
            fmt=lambda v: f"{v:.1f} Hz", command=self._on_pitch,
            init=settings.pitch_default_hz, step=5)
        self._pitch.grid(row=0, column=0, sticky="ew")

        # ── wave shape ──────────────────────────────────────────────────────
        wave = Section(body, "Waveform  (click to edit)")
        wave.grid(row=4, column=0, sticky="ew")
        self._wave_viewer = WaveViewer(
            wave.content, engine.get_wave_params(), on_click=self._open_wave_editor)
        self._wave_viewer.pack(anchor="w")
        ttk.Label(wave.content, text="POST FILTER", style="Section.TLabel").pack(
            anchor="w", pady=(6, 2))
        self._filtered_viewer = WaveViewer(
            wave.content, engine.get_filtered_wave_params())
        self._filtered_viewer.pack(anchor="w")
        self._wave_editor = None

        # ── synth voice: envelopes (amp | filter, side by side) ─────────────
        envs = Section(body, "Envelopes  (amp / filter)")
        envs.grid(row=5, column=0, sticky="ew")
        self._build_adsr(envs.content, "amp_env", column=0)
        ttk.Separator(envs.content, orient="vertical").grid(
            row=0, column=1, sticky="ns", padx=8)
        self._build_adsr(envs.content, "filter_env", column=2)

        # ── synth voice: filter (same bar geometry as the envelopes) ────────
        filt = Section(body, "Filter  (resonant low-pass)")
        filt.grid(row=6, column=0, sticky="ew")
        self._build_filter_bars(filt.content)

        # ── advanced DSP (hideable) ─────────────────────────────────────────
        self._dsp_pane = DspSettingsPane(body, settings, self._on_apply_dsp)
        self._dsp_pane.grid(row=7, column=0, sticky="ew")

        self._update_kind_state()
        self._poll_position()

    # ── synth voice widgets ─────────────────────────────────────────────────
    def _build_adsr(self, parent, prefix: str, column: int = 0) -> None:
        """Compact A/D/S/R fill-bar bank (wave-editor style) writing straight
        into the settings group; the voice reads them live at every chunk
        boundary, so no engine call. Bar values are normalized 0..1: times
        map to 0..env_time_max_s, sustain is the fraction itself."""
        s = self._settings
        t_max = s.env_time_max_s
        ms = lambda v: f"{v * t_max * 1000:.0f}ms"
        pct = lambda v: f"{v * 100:.0f}%"

        def time_setter(attr):
            return lambda v: setattr(self._settings, attr, v * t_max)

        specs = (
            ("A", getattr(s, f"{prefix}_attack_s") / t_max, ms,
             time_setter(f"{prefix}_attack_s")),
            ("D", getattr(s, f"{prefix}_decay_s") / t_max, ms,
             time_setter(f"{prefix}_decay_s")),
            ("S", getattr(s, f"{prefix}_sustain"), pct,
             lambda v: setattr(self._settings, f"{prefix}_sustain", v)),
            ("R", getattr(s, f"{prefix}_release_s") / t_max, ms,
             time_setter(f"{prefix}_release_s")),
        )
        VBarBank(parent, specs).grid(row=0, column=column, sticky="w")

    def _build_filter_bars(self, parent) -> None:
        """Cutoff/resonance as fill bars matching the envelope banks, plus the
        bipolar env-amount bar (grows from the centre, left = negative)."""
        s = self._settings
        log_lo = math.log10(s.filter_min_cutoff_hz)
        log_hi = math.log10(s.fs_out / 2.0)
        q_lo, q_hi = 0.5, 10.0

        def cutoff_hz(norm):
            return 10.0 ** (log_lo + norm * (log_hi - log_lo))

        specs = (
            ("Cut", (math.log10(s.filter_cutoff_hz) - log_lo) / (log_hi - log_lo),
             lambda v: f"{cutoff_hz(v):.0f}Hz",
             lambda v: setattr(self._settings, "filter_cutoff_hz", cutoff_hz(v))),
            ("Res", (s.filter_resonance_q - q_lo) / (q_hi - q_lo),
             lambda v: f"Q{q_lo + v * (q_hi - q_lo):.1f}",
             lambda v: setattr(self._settings, "filter_resonance_q",
                               q_lo + v * (q_hi - q_lo))),
        )
        VBarBank(parent, specs, col_w=56).grid(row=0, column=0, sticky="w")

        amount = ttk.Frame(parent)
        amount.grid(row=0, column=1, sticky="n", padx=(16, 0))
        ttk.Label(amount, text="ENV AMT", style="Section.TLabel").pack(anchor="w")
        bar = BipolarBar(
            amount, s.filter_env_amount, fmt=lambda v: f"{v * 100:+.0f}%",
            on_change=lambda v: setattr(self._settings, "filter_env_amount", v))
        bar.pack(anchor="w", pady=(4, 2))
        readout = ttk.Label(amount, style="Value.TLabel")
        readout.pack(anchor="w")
        bar.set_readout(lambda text: readout.configure(text=text))

    # ── handlers ───────────────────────────────────────────────────────────
    def _on_device_selected(self, _event=None) -> None:
        self._engine.set_output_device(self.selected_output_device())
        if self._on_device_change is not None:
            self._on_device_change()

    def selected_output_device(self):
        name = self._device_var.get()
        for idx, dev_name in self._devices:
            if dev_name == name:
                return idx
        return None

    @staticmethod
    def _find_cable_device_name(names) -> Optional[str]:
        for name in names:
            if "cable" in name.lower():
                return name
        return None

    def selected_output_device2(self):
        name = self._device2_var.get()
        for idx, dev_name in self._devices:
            if dev_name == name:
                return idx
        return None

    def _on_output2_toggle(self) -> None:
        enabled = self._out2_enabled_var.get()
        self._combo2.configure(state="readonly" if enabled else "disabled")
        self._engine.set_output2_enabled(enabled)
        self._engine.set_output_device2(self.selected_output_device2())
        if self._on_device_change is not None:
            self._on_device_change()

    def _on_device2_selected(self, _event=None) -> None:
        self._engine.set_output_device2(self.selected_output_device2())
        if self._on_device_change is not None:
            self._on_device_change()

    def _on_kind(self, kind: str) -> None:
        self._engine.set_payload_kind(kind)
        self._file_row.set(os.path.basename(self._engine.get_payload_path(kind) or ""))
        self._update_kind_state()
        if self._on_kind_change is not None:
            self._on_kind_change(kind)

    def _on_strategy(self, kind: str) -> None:
        self._engine.set_strategy_kind(kind)
        if self._on_strategy_change is not None:
            self._on_strategy_change(kind)

    def _on_codec(self, value: str) -> None:
        mode = SerializerMode.DIGITAL if value == "digital" else SerializerMode.ANALOGUE
        self._engine.set_codec_mode(mode)

    def _on_bits(self, _event=None) -> None:
        try:
            self._engine.set_bits_per_symbol(int(self._bits_var.get()))
        except Exception as exc:
            messagebox.showerror("Bits per Symbol Error", str(exc))
            self._bits_var.set(str(self._settings.bits_per_symbol))

    def _on_samples_per_symbol(self, _event=None) -> None:
        try:
            self._engine.set_audio_samples_per_symbol(int(self._samples_var.get()))
        except Exception as exc:
            messagebox.showerror("Samples per Symbol Error", str(exc))
            self._samples_var.set(str(self._settings.audio_samples_per_symbol))

    def _on_pick_payload(self) -> None:
        title, filetypes = _PAYLOAD_DIALOGS[self._kind_seg.get()]
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if not file_path:
            return
        try:
            self._engine.load_payload_file(file_path)
        except Exception as exc:
            messagebox.showerror("Payload Error", str(exc))
            return
        self._file_row.set(os.path.basename(file_path))

    def _on_position_press(self, _event) -> None:
        self._payload_dragging = True

    def _on_position_release(self, _event) -> None:
        fraction = self._position.get() / self._settings.position_slider_max
        self._engine.set_position_fraction(fraction)
        self._payload_dragging = False

    def _on_pitch(self, f0: float) -> None:
        self._settings.pitch_default_hz = f0
        self._engine.set_encoder_f0(f0)
        if self._on_pitch_change is not None:
            self._on_pitch_change(f0)

    def _open_wave_editor(self) -> None:
        if self._wave_editor is not None and self._wave_editor.winfo_exists():
            self._wave_editor.lift()
            self._wave_editor.focus_set()
            return
        self._wave_editor = WaveEditorWindow(
            self.winfo_toplevel(), self._settings,
            self._engine.get_wave_params(), self._on_wave_change)

    def _on_wave_change(self, params) -> None:
        try:
            self._engine.set_wave_params(params)
        except Exception as exc:
            messagebox.showerror("Wave Error", str(exc))
            return
        self._wave_viewer.set_params(params)

    def _on_apply_dsp(self, values: dict) -> None:
        self._engine.apply_dsp_settings(values)
        # Wave params may have been reset to the (possibly re-sized) default;
        # an open editor would keep pushing stale-length params, so close it.
        if self._wave_editor is not None and self._wave_editor.winfo_exists():
            self._wave_editor.destroy()
        self._wave_editor = None
        self._wave_viewer.set_params(self._engine.get_wave_params())
        # chunk_size may have changed; reuse the strategy hook so a linked
        # decoder panel refreshes its offset slider range.
        if self._on_strategy_change is not None:
            self._on_strategy_change(self._strategy_seg.get())

    def _update_kind_state(self) -> None:
        self._samples_combo.configure(
            state="readonly" if self._kind_seg.get() == "audio" else "disabled")

    # ── notified by the keyboard bar / app ─────────────────────────────────
    def set_note_control_active(self, active: bool, f0: float) -> None:
        """While notes drive the pitch, the slider follows and is locked."""
        self._pitch.set_enabled(not active)
        if active:
            self._pitch.set_silent(min(max(f0, self._settings.pitch_min_hz),
                                       self._settings.pitch_max_hz))
            self._pitch.set_readout(f"{f0:.1f} Hz")

    def get_pitch(self) -> float:
        return self._pitch.get()

    # ── polling ────────────────────────────────────────────────────────────
    def _poll_position(self) -> None:
        if not self._alive:
            return
        if not self._payload_dragging:
            fraction = self._engine.get_position_fraction()
            self._position.set_silent(fraction * self._settings.position_slider_max)
        # Live post-filter view: redraws as the filter envelope sweeps the
        # cutoff while playing (and tracks slider moves while idle).
        self._filtered_viewer.set_params(self._engine.get_filtered_wave_params())
        self.after(self._settings.gui_poll_interval_ms, self._poll_position)

    def destroy(self) -> None:
        self._alive = False
        super().destroy()
