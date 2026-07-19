"""Decoder rack module: tuning, reconstruction and decoded output views."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from PIL import Image as PILImage, ImageTk

from gui.settings_pane import DspSettingsPane
from gui.theme import Palette
from gui.widgets import (LabeledScale, Panel, Section, Segmented,
                         make_text_well, set_text_well)
from SerializerMode import SerializerMode
from Settings import Settings
from Sink import SinkBehaviour
from WaveParams import WaveParams

_F0_MODES = {"Manual": "manual", "Autocorrelation": "autocorr", "FFT": "fft"}
_RESAMPLE_METHODS = {
    "Polyphase": "poly",
    "Linear": "linear",
    "Zero-order hold": "hold",
}


class DecoderPanel(Panel):
    """UI for the decoder side. Works against DecoderEngine or LinkedEngine.

    In linked mode the transmission parameters and audio devices are owned by
    the encoder panel, so those sections are not shown here.
    """

    def __init__(self, parent, engine, settings: Settings, *,
                 linked: bool,
                 input_devices=(), output_devices=(),
                 on_close: Callable[[], None] = None,
                 on_device_change: Optional[Callable[[], None]] = None,
                 get_encoder_pitch: Optional[Callable[[], float]] = None):
        super().__init__(parent, "DECODER", on_close=on_close)
        self._engine = engine
        self._settings = settings
        self._linked = linked
        self._on_device_change = on_device_change
        self._get_encoder_pitch = get_encoder_pitch
        self._input_devices = input_devices
        self._output_devices = output_devices
        self._alive = True

        self._pending_image_frame = None
        self._pending_raw_frame = None
        self._pending_text: Optional[str] = None
        self._pending_raw_text: Optional[str] = None
        self._preview_photo = None
        self._raw_preview_photo = None

        engine.set_on_image(self._cb_image)
        engine.set_on_raw_image(self._cb_raw_image)
        engine.set_on_data(self._cb_data)
        engine.set_on_raw_data(self._cb_raw_data)
        engine.set_on_text(self._cb_text)
        engine.set_on_raw_text(self._cb_raw_text)

        # signal readout lives in the module header, like a channel LED strip
        self._signal_var = tk.StringVar(value="")
        self._signal_label = ttk.Label(self.header_extra, textvariable=self._signal_var,
                                       style="Title.TLabel")
        self._signal_label.pack(side="left")

        body = self.body
        body.columnconfigure(0, weight=1)
        row = 0

        # ── devices (standalone only) ───────────────────────────────────────
        if not linked:
            dev = Section(body, "Audio Devices")
            dev.grid(row=row, column=0, sticky="ew"); row += 1
            dev.content.columnconfigure(1, weight=1)

            ttk.Label(dev.content, text="Input", style="Dim.TLabel", width=9).grid(
                row=0, column=0, sticky="w", pady=2)
            in_names = ["(default)"] + [n for _, n in input_devices]
            self._in_var = tk.StringVar(value=in_names[0])
            in_combo = ttk.Combobox(dev.content, textvariable=self._in_var,
                                    values=in_names, state="readonly")
            in_combo.grid(row=0, column=1, sticky="ew", pady=2)
            in_combo.bind("<<ComboboxSelected>>", self._on_device_selected)

            ttk.Label(dev.content, text="Output", style="Dim.TLabel", width=9).grid(
                row=1, column=0, sticky="w", pady=2)
            out_names = ["(default)"] + [n for _, n in output_devices]
            self._out_var = tk.StringVar(value=out_names[0])
            out_combo = ttk.Combobox(dev.content, textvariable=self._out_var,
                                     values=out_names, state="readonly")
            out_combo.grid(row=1, column=1, sticky="ew", pady=2)
            out_combo.bind("<<ComboboxSelected>>", self._on_device_selected)

            # ── transmission parameters (standalone only) ───────────────────
            tx = Section(body, "Transmission")
            tx.grid(row=row, column=0, sticky="ew"); row += 1
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

            ttk.Label(tx.content, text="Img codec", style="Dim.TLabel", width=9).grid(
                row=2, column=0, sticky="w", pady=2)
            self._codec_seg = Segmented(
                tx.content, [("Digital", "digital"), ("Analogue", "analogue")],
                self._on_codec, "digital")
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
        else:
            self._kind_seg = None
            link = ttk.Label(body, text="⛓  Transmission parameters linked to encoder",
                             style="Dim.TLabel")
            link.grid(row=row, column=0, sticky="w", pady=(6, 0)); row += 1

        # ── tuning ──────────────────────────────────────────────────────────
        tune = Section(body, "Tuning")
        tune.grid(row=row, column=0, sticky="ew"); row += 1
        tune.content.columnconfigure(1, weight=1)

        mode_row = ttk.Frame(tune.content)
        mode_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(mode_row, text="f0 mode", style="Dim.TLabel", width=9).pack(side="left")
        self._f0_mode_var = tk.StringVar(value="Manual")
        f0_combo = ttk.Combobox(mode_row, textvariable=self._f0_mode_var,
                                values=list(_F0_MODES), state="readonly", width=14)
        f0_combo.pack(side="left")
        f0_combo.bind("<<ComboboxSelected>>", self._on_f0_mode)

        self._quantize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_row, text="Quantize pitch", variable=self._quantize_var,
                        command=self._on_quantize).pack(side="left", padx=12)

        self._pitch = LabeledScale(
            tune.content, "Pitch", settings.pitch_min_hz, settings.pitch_max_hz,
            fmt=lambda v: f"{v:.1f} Hz", command=self._on_pitch,
            init=settings.pitch_default_hz, step=5)
        self._pitch.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

        if linked:
            self._follow_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(tune.content, text="Follow encoder pitch",
                            variable=self._follow_var,
                            command=self._on_follow_change).grid(
                row=2, column=0, columnspan=2, sticky="w", pady=2)
        else:
            self._follow_var = None

        self._tune_offset = LabeledScale(
            tune.content, "Offset", 0, settings.chunk_size - 1,
            fmt=lambda v: f"{int(v)}", command=self._on_tune_offset, init=0)
        self._tune_offset.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)

        resample_row = ttk.Frame(tune.content)
        resample_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(resample_row, text="Upsample", style="Dim.TLabel", width=9).pack(side="left")
        self._resample_var = tk.StringVar(value="Polyphase")
        resample_combo = ttk.Combobox(resample_row, textvariable=self._resample_var,
                                      values=list(_RESAMPLE_METHODS),
                                      state="readonly", width=14)
        resample_combo.pack(side="left")
        resample_combo.bind("<<ComboboxSelected>>", self._on_resample)

        # ── carrier wave (standalone only: linked mode syncs it from the
        #    encoder's wave editor automatically) ─────────────────────────────
        if not linked:
            wave_row = ttk.Frame(tune.content)
            wave_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=2)
            ttk.Label(wave_row, text="Wave", style="Dim.TLabel", width=9).pack(side="left")
            ttk.Button(wave_row, text="Import wave…",
                       command=self._on_import_wave).pack(side="left")
            self._wave_var = tk.StringVar(value="harmonic (default)")
            ttk.Label(wave_row, textvariable=self._wave_var,
                      style="Value.TLabel").pack(side="left", padx=8)

        # ── reconstruction ──────────────────────────────────────────────────
        recon = Section(body, "Reconstruction")
        recon.grid(row=row, column=0, sticky="ew"); row += 1
        recon.content.columnconfigure(1, weight=1)
        ttk.Label(recon.content, text="Mode", style="Dim.TLabel", width=9).grid(
            row=0, column=0, sticky="w")
        self._sink_seg = Segmented(recon.content, [("Live", "live"), ("Clean", "clean")],
                                   self._on_sink, "live")
        self._sink_seg.grid(row=0, column=1, sticky="ew")

        # ── output ──────────────────────────────────────────────────────────
        out = Section(body, "Decoded Output")
        out.grid(row=row, column=0, sticky="ew"); row += 1
        out.content.columnconfigure(0, weight=1)
        out.content.columnconfigure(1, weight=1)

        ttk.Label(out.content, text="Synced / Clean", style="Dim.TLabel").grid(
            row=0, column=0)
        ttk.Label(out.content, text="Raw (no sync)", style="Dim.TLabel").grid(
            row=0, column=1)

        size = settings.image_preview_size
        self._preview = tk.Label(out.content, text="(no frame)", bg=Palette.INSET,
                                 fg=Palette.DIM, width=1, height=1)
        self._raw_preview = tk.Label(out.content, text="(no frame)", bg=Palette.INSET,
                                     fg=Palette.DIM, width=1, height=1)
        for col, label in ((0, self._preview), (1, self._raw_preview)):
            label.configure(font=Palette.FONT)
            label.grid(row=1, column=col, sticky="nsew",
                       padx=(0, 6) if col == 0 else 0, pady=(2, 4),
                       ipadx=size // 2, ipady=size // 2)

        self._text_well = make_text_well(out.content, width=22, height=6)
        self._raw_text_well = make_text_well(out.content, width=22, height=6)
        self._text_well.grid(row=2, column=0, sticky="nsew", padx=(0, 6))
        self._raw_text_well.grid(row=2, column=1, sticky="nsew")

        btn_row = ttk.Frame(out.content)
        btn_row.grid(row=3, column=0, columnspan=2, pady=(8, 0))
        self._save_audio_btn = ttk.Button(btn_row, text="Save audio (WAV)…",
                                          command=self._on_save_audio)
        self._save_audio_btn.pack(side="left", padx=(0, 8))
        self._save_data_btn = ttk.Button(btn_row, text="Save clean output…",
                                         command=self._on_save_data)
        self._save_data_btn.pack(side="left")

        # ── advanced DSP (hideable; linked mode edits it on the encoder) ────
        if not linked:
            self._dsp_pane = DspSettingsPane(body, settings, self._on_apply_dsp)
            self._dsp_pane.grid(row=row, column=0, sticky="ew"); row += 1

        self._current_kind = "audio"
        self._update_kind_state()
        self._update_f0_mode_state()
        self._poll_image()
        self._poll_text()
        self._poll_signal()

    # ── device selection (standalone) ───────────────────────────────────────
    def _selected(self, var: tk.StringVar, devices):
        name = var.get()
        for idx, dev_name in devices:
            if dev_name == name:
                return idx
        return None

    def apply_devices(self) -> None:
        if self._linked:
            return
        self._engine.set_input_device(self._selected(self._in_var, self._input_devices))
        self._engine.set_output_device(self._selected(self._out_var, self._output_devices))

    def _on_device_selected(self, _event=None) -> None:
        self.apply_devices()
        if self._on_device_change is not None:
            self._on_device_change()

    # ── transmission handlers (standalone) ──────────────────────────────────
    def _on_kind(self, kind: str) -> None:
        self._engine.set_payload_kind(kind)
        self.on_payload_kind_changed(kind)

    def _on_strategy(self, kind: str) -> None:
        self._engine.set_strategy_kind(kind)
        self.on_strategy_changed()

    def _on_codec(self, value: str) -> None:
        mode = SerializerMode.DIGITAL if value == "digital" else SerializerMode.ANALOGUE
        self._engine.set_codec_mode(mode)

    def _on_bits(self, _event=None) -> None:
        try:
            self._engine.set_bits_per_symbol(int(self._bits_var.get()))
        except Exception as exc:
            messagebox.showerror("Bits per Symbol Error", str(exc))
            self._bits_var.set(str(self._settings.bits_per_symbol))

    def _on_apply_dsp(self, values: dict) -> None:
        prev_total = self._settings.total_harmonics
        self._engine.apply_dsp_settings(values)
        self.on_strategy_changed()
        # A harmonic-count change drops any imported carrier scalars.
        if self._settings.total_harmonics != prev_total:
            self._wave_var.set("harmonic (default)")

    # ── notifications from the app (linked mode) ────────────────────────────
    def on_strategy_changed(self) -> None:
        """chunk_size follows the strategy; the offset slider range must too."""
        self._tune_offset.set_range(0, self._settings.chunk_size - 1)
        self._tune_offset.set_silent(self._engine.get_tune_offset())

    def on_payload_kind_changed(self, kind: str) -> None:
        self._current_kind = kind
        self._pending_image_frame = self._pending_raw_frame = None
        self._preview_photo = self._raw_preview_photo = None
        self._preview.configure(image="", text="(no frame)")
        self._raw_preview.configure(image="", text="(no frame)")
        set_text_well(self._text_well, "(no data yet)")
        set_text_well(self._raw_text_well, "(no data yet)")
        self._update_kind_state()

    def on_encoder_pitch_changed(self, f0: float) -> None:
        """Linked mode: encoder slider moved while 'follow' is on."""
        if self._follow_var is not None and self._follow_var.get():
            self._pitch.set_silent(f0)
            self._engine.set_decoder_f0(f0)

    # ── tuning handlers ─────────────────────────────────────────────────────
    def _on_pitch(self, f0: float) -> None:
        self._engine.set_decoder_f0(f0)

    def _on_follow_change(self) -> None:
        self._update_f0_mode_state()
        if self._follow_var.get() and self._get_encoder_pitch is not None:
            f0 = self._get_encoder_pitch()
            self._pitch.set_silent(f0)
            self._engine.set_decoder_f0(f0)

    def _on_f0_mode(self, _event=None) -> None:
        self._engine.set_f0_estimator_mode(_F0_MODES[self._f0_mode_var.get()])
        self._update_f0_mode_state()

    def _on_quantize(self) -> None:
        self._engine.set_pitch_quantize(self._quantize_var.get())

    def _on_tune_offset(self, value: float) -> None:
        self._engine.set_tune_offset(int(value))

    def _on_resample(self, _event=None) -> None:
        self._engine.set_resample_method(_RESAMPLE_METHODS[self._resample_var.get()])

    def _on_import_wave(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Import wave settings (exported from the encoder)",
            filetypes=[("SSF wave files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            params = WaveParams.from_json_file(file_path)
            if len(params.omegas) != self._settings.total_harmonics:
                raise ValueError(
                    f"File carries {len(params.omegas)} harmonics, this rack "
                    f"runs {self._settings.total_harmonics}")
            self._engine.set_harmonic_scalars(params.omegas)
        except Exception as exc:
            messagebox.showerror("Import Wave Error", str(exc))
            return
        self._wave_var.set(os.path.basename(file_path))

    def _on_sink(self, value: str) -> None:
        behaviour = SinkBehaviour.LIVE if value == "live" else SinkBehaviour.CLEAN
        self._engine.set_sink_behaviour(behaviour)

    def _update_f0_mode_state(self) -> None:
        # Manual pitch only makes sense when no estimator drives it and (in
        # linked mode) the slider is not slaved to the encoder.
        manual = self._f0_mode_var.get() == "Manual"
        following = self._follow_var is not None and self._follow_var.get()
        self._pitch.set_enabled(manual and not following)

    def _update_kind_state(self) -> None:
        kind = self._current_kind
        if self._kind_seg is not None:
            self._codec_seg.set_enabled(kind != "audio")
        self._sink_seg.set_enabled(kind != "audio")
        self._save_audio_btn.state(["!disabled" if kind == "audio" else "disabled"])
        self._save_data_btn.state(
            ["!disabled" if kind in ("binary", "text") else "disabled"])

    # ── save actions ────────────────────────────────────────────────────────
    def _on_save_audio(self) -> None:
        file_path = filedialog.asksaveasfilename(
            title="Save decoded audio", defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            self._engine.dump_decoded_audio_to_wav(file_path)
        except Exception as exc:
            messagebox.showerror("Save Audio Error", str(exc))

    def _on_save_data(self) -> None:
        kind = self._current_kind
        file_path = filedialog.asksaveasfilename(
            title="Save decoded data",
            defaultextension=".txt" if kind == "text" else "",
            filetypes=[("All files", "*.*")])
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

    # ── decoder callbacks (audio thread → Tk poll) ──────────────────────────
    def _cb_image(self, frame) -> None:
        self._pending_image_frame = frame

    def _cb_raw_image(self, frame) -> None:
        self._pending_raw_frame = frame

    def _cb_data(self, data: bytes) -> None:
        self._pending_text = f"{len(data)} bytes decoded"

    def _cb_raw_data(self, data: bytes) -> None:
        self._pending_raw_text = f"...{data.hex()[-400:]}"

    def _cb_text(self, text: str) -> None:
        self._pending_text = text

    def _cb_raw_text(self, text: str) -> None:
        self._pending_raw_text = text

    def _render_frame(self, frame, label: tk.Label):
        pixels, width, height, channels = frame
        mode = "L" if channels == 1 else "RGB"
        try:
            image = PILImage.frombytes(mode, (width, height), bytes(pixels))
            size = self._settings.image_preview_size
            image = image.resize((size, size), PILImage.NEAREST)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo, text="")
            return photo
        except Exception:
            return None

    def _poll_image(self) -> None:
        if not self._alive:
            return
        frame = self._pending_image_frame
        if frame is not None:
            self._pending_image_frame = None
            photo = self._render_frame(frame, self._preview)
            if photo is not None:
                self._preview_photo = photo
        raw_frame = self._pending_raw_frame
        if raw_frame is not None:
            self._pending_raw_frame = None
            photo = self._render_frame(raw_frame, self._raw_preview)
            if photo is not None:
                self._raw_preview_photo = photo
        self.after(self._settings.gui_poll_interval_ms, self._poll_image)

    def _poll_text(self) -> None:
        if not self._alive:
            return
        if self._pending_text is not None:
            set_text_well(self._text_well, self._pending_text)
            self._pending_text = None
        if self._pending_raw_text is not None:
            set_text_well(self._raw_text_well, self._pending_raw_text)
            self._pending_raw_text = None
        self.after(self._settings.gui_poll_interval_ms, self._poll_text)

    def _poll_signal(self) -> None:
        if not self._alive:
            return
        is_manual = self._f0_mode_var.get() == "Manual"
        if not is_manual:
            f0 = self._engine.get_estimated_f0()
            if f0 > 0.0:
                self._pitch.set_silent(min(max(f0, self._settings.pitch_min_hz),
                                           self._settings.pitch_max_hz))
                self._pitch.set_readout(f"{f0:.1f} Hz")

        drop_run = self._engine.get_drop_run()
        if self._engine.is_gated() or drop_run > 0:
            self._signal_label.configure(foreground=Palette.RED)
            self._signal_var.set(
                f"SIGNAL WEAK  {drop_run}/{self._settings.drop_tolerance_chunks}")
        elif not is_manual:
            self._signal_label.configure(foreground=Palette.GREEN)
            self._signal_var.set(f"conf {self._engine.get_confidence():.2f}")
        else:
            self._signal_var.set("")
        self.after(self._settings.gui_poll_interval_ms, self._poll_signal)

    def destroy(self) -> None:
        self._alive = False
        super().destroy()
