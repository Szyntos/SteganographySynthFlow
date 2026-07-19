"""SSF rack window.

Starts as an empty rack; the user adds an Encoder and/or Decoder module.
With both present the modules share one audio stream (encoder feeds the
decoder internally) and their transmission parameters are linked; with one
present it runs standalone against real audio devices. A piano keyboard bar
spans the bottom whenever an encoder is racked — think of it as a real synth.

Run a second instance of the program for an independent encoder/decoder pair.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # Settings uses paths relative to the project root

from gui.decoder_panel import DecoderPanel
from gui.encoder_panel import EncoderPanel
from gui.engines import (DecoderEngine, EncoderEngine, LinkedEngine,
                         list_audio_devices)
from gui.keyboard_bar import KeyboardBar
from gui.theme import Palette, apply_theme
from gui.widgets import Led, ScrollFrame, Segmented
from Settings import Settings


class RackApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SteganographySynthFlow")
        apply_theme(self)

        settings = Settings()
        settings.validate()
        self._settings = settings

        self._engine = None
        self._encoder_panel = None
        self._decoder_panel = None
        self._keyboard_bar = None
        self._running = False

        self._input_devices = list_audio_devices("input")
        self._output_devices = list_audio_devices("output")

        self._build_transport_bar()
        self._rack_scroll = ScrollFrame(self)
        self._rack_scroll.grid(row=1, column=0, sticky="nsew")
        self._rack = ttk.Frame(self._rack_scroll.inner, style="Rack.TFrame", padding=10)
        self._rack.pack(fill="both", expand=True)
        self.minsize(760, 420)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._keyboard_slot = ttk.Frame(self, style="Bar.TFrame")
        self._keyboard_slot.grid(row=2, column=0, sticky="ew")

        self._rebuild_rack()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── transport bar ───────────────────────────────────────────────────────
    def _build_transport_bar(self) -> None:
        bar = ttk.Frame(self, style="Bar.TFrame", padding=(12, 8))
        bar.grid(row=0, column=0, sticky="ew")

        ttk.Label(bar, text="SSF", style="Bar.TLabel",
                  font=("Segoe UI Black", 13)).pack(side="left")
        ttk.Label(bar, text="SteganographySynthFlow", style="Bar.TLabel",
                  font=Palette.FONT_SMALL).pack(side="left", padx=(8, 20))

        self._toggle_btn = ttk.Button(bar, text="▶  Start", style="Accent.TButton",
                                      command=self._toggle)
        self._toggle_btn.pack(side="left")

        self._led = Led(bar)
        self._led.pack(side="left", padx=(12, 4))
        self._status_var = tk.StringVar(value="Stopped")
        ttk.Label(bar, textvariable=self._status_var, style="Bar.TLabel").pack(side="left")

        # right-hand side: add-module buttons, master volume, monitor source
        self._add_dec_btn = ttk.Button(bar, text="＋ Decoder",
                                       command=lambda: self._add_module("decoder"))
        self._add_dec_btn.pack(side="right", padx=(6, 0))
        self._add_enc_btn = ttk.Button(bar, text="＋ Encoder",
                                       command=lambda: self._add_module("encoder"))
        self._add_enc_btn.pack(side="right")

        vol_frame = ttk.Frame(bar, style="Bar.TFrame")
        vol_frame.pack(side="right", padx=16)
        ttk.Label(vol_frame, text="Vol", style="Bar.TLabel").pack(side="left", padx=(0, 6))
        self._vol_scale = ttk.Scale(vol_frame, from_=self._settings.volume_min_db,
                                    to=self._settings.volume_max_db,
                                    orient="horizontal", length=130,
                                    command=self._on_volume)
        self._vol_scale.pack(side="left")
        self._vol_label = ttk.Label(vol_frame, text="", width=7, anchor="e",
                                    style="Bar.TLabel", font=Palette.FONT_MONO)
        self._vol_label.pack(side="left")
        self._vol_scale.set(self._settings.volume_default_db)

        self._monitor_frame = ttk.Frame(bar, style="Bar.TFrame")
        ttk.Label(self._monitor_frame, text="Monitor", style="Bar.TLabel").pack(
            side="left", padx=(0, 6))
        self._monitor_seg = Segmented(
            self._monitor_frame, [("Enc", "encoder"), ("Dec", "decoder")],
            self._on_monitor, "encoder")
        self._monitor_seg.pack(side="left")

    # ── module management ───────────────────────────────────────────────────
    def _add_module(self, which: str) -> None:
        self._rebuild_rack(add=which)

    def _remove_module(self, which: str) -> None:
        self._rebuild_rack(remove=which)

    def _rebuild_rack(self, add: str = None, remove: str = None) -> None:
        has_enc = self._encoder_panel is not None
        has_dec = self._decoder_panel is not None
        if add == "encoder":
            has_enc = True
        if add == "decoder":
            has_dec = True
        if remove == "encoder":
            has_enc = False
        if remove == "decoder":
            has_dec = False

        # tear down what exists (audio first, then UI)
        if self._running:
            self._stop()
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        for widget in (self._keyboard_bar, self._encoder_panel, self._decoder_panel):
            if widget is not None:
                widget.destroy()
        self._keyboard_bar = self._encoder_panel = self._decoder_panel = None
        for child in self._rack.winfo_children():
            child.destroy()

        linked = has_enc and has_dec
        if linked:
            self._engine = LinkedEngine(self._settings)
        elif has_enc:
            self._engine = EncoderEngine(self._settings)
        elif has_dec:
            self._engine = DecoderEngine(self._settings)

        if not has_enc and not has_dec:
            self._build_empty_state()
        else:
            col = 0
            self._rack.rowconfigure(0, weight=1)
            if has_enc:
                self._encoder_panel = EncoderPanel(
                    self._rack, self._engine, self._settings,
                    linked=linked, output_devices=self._output_devices,
                    on_close=lambda: self._remove_module("encoder"),
                    on_strategy_change=self._on_encoder_strategy_change,
                    on_kind_change=self._on_encoder_kind_change,
                    on_device_change=self._restart_if_running,
                    on_pitch_change=self._on_encoder_pitch_change)
                self._encoder_panel.grid(row=0, column=col, sticky="nsew",
                                         padx=(0, 10 if has_dec else 0))
                self._rack.columnconfigure(col, weight=1)
                col += 1
            if has_dec:
                self._decoder_panel = DecoderPanel(
                    self._rack, self._engine, self._settings,
                    linked=linked,
                    input_devices=self._input_devices,
                    output_devices=self._output_devices,
                    on_close=lambda: self._remove_module("decoder"),
                    on_device_change=self._restart_if_running,
                    get_encoder_pitch=(
                        (lambda: self._encoder_panel.get_pitch()) if linked else None))
                self._decoder_panel.grid(row=0, column=col, sticky="nsew")
                self._rack.columnconfigure(col, weight=1)

            if has_enc:
                self._keyboard_bar = KeyboardBar(
                    self._keyboard_slot, self._engine, self._settings,
                    on_note_pitch=self._on_note_pitch)
                self._keyboard_bar.pack(fill="x")

        # transport-bar state follows the module set
        self._add_enc_btn.state(["disabled" if has_enc else "!disabled"])
        self._add_dec_btn.state(["disabled" if has_dec else "!disabled"])
        self._toggle_btn.state(["!disabled" if self._engine is not None else "disabled"])
        if linked:
            self._monitor_frame.pack(side="right", padx=(0, 4))
            self._monitor_seg.set_silent(self._engine.get_source())
        else:
            self._monitor_frame.pack_forget()
        self._on_volume(self._vol_scale.get())

    def _build_empty_state(self) -> None:
        empty = ttk.Frame(self._rack, style="Rack.TFrame")
        empty.place(relx=0.5, rely=0.5, anchor="center")
        ttk.Label(empty, text="Empty rack", style="Bar.TLabel",
                  font=("Segoe UI", 14)).pack(pady=(0, 4))
        ttk.Label(empty, text="Add a module to begin", style="Bar.TLabel",
                  foreground=Palette.DIM).pack(pady=(0, 14))
        row = ttk.Frame(empty, style="Rack.TFrame")
        row.pack()
        ttk.Button(row, text="＋ Encoder", style="Accent.TButton",
                   command=lambda: self._add_module("encoder")).pack(side="left", padx=6)
        ttk.Button(row, text="＋ Decoder", style="Accent.TButton",
                   command=lambda: self._add_module("decoder")).pack(side="left", padx=6)

    # ── cross-panel wiring ──────────────────────────────────────────────────
    def _on_encoder_strategy_change(self, _kind: str) -> None:
        if self._decoder_panel is not None:
            self._decoder_panel.on_strategy_changed()

    def _on_encoder_kind_change(self, kind: str) -> None:
        if self._decoder_panel is not None:
            self._decoder_panel.on_payload_kind_changed(kind)

    def _on_encoder_pitch_change(self, f0: float) -> None:
        if self._decoder_panel is not None:
            self._decoder_panel.on_encoder_pitch_changed(f0)

    def _on_note_pitch(self, gate_active: bool, f0: float) -> None:
        if self._encoder_panel is not None:
            self._encoder_panel.set_note_control_active(gate_active, f0)
        if (self._decoder_panel is not None and gate_active
                and self._encoder_panel is not None):
            self._decoder_panel.on_encoder_pitch_changed(f0)

    # ── transport ───────────────────────────────────────────────────────────
    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        if self._engine is None:
            return
        if self._decoder_panel is not None:
            self._decoder_panel.apply_devices()
        try:
            self._engine.start()
        except Exception as exc:
            messagebox.showerror("Audio Error", str(exc))
            return
        self._running = True
        self._status_var.set("Running")
        self._led.set_color(Palette.GREEN)
        self._toggle_btn.configure(text="⏹  Stop")

    def _stop(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        self._running = False
        self._status_var.set("Stopped")
        self._led.set_color(Palette.PANEL_EDGE)
        self._toggle_btn.configure(text="▶  Start")

    def _restart_if_running(self) -> None:
        if self._running:
            self._engine.stop()
            self._start()

    def _on_monitor(self, source: str) -> None:
        self._engine.set_source(source)

    def _on_volume(self, value) -> None:
        db = float(value)
        self._settings.volume_default_db = db
        gain = 0.0 if db <= self._settings.volume_min_db else 10 ** (db / 20.0)
        if self._engine is not None:
            self._engine.set_volume(gain)
        label = "−∞ dB" if db <= self._settings.volume_min_db else f"{db:.0f} dB"
        self._vol_label.configure(text=label)

    def _on_close(self) -> None:
        if self._keyboard_bar is not None:
            self._keyboard_bar.shutdown()
        if self._engine is not None:
            self._engine.shutdown()
        self.destroy()


def main() -> None:
    app = RackApp()
    app.mainloop()


if __name__ == "__main__":
    main()
