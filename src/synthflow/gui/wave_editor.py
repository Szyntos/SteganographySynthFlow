"""Additive wave editor: the small clickable cycle display that sits in the
encoder panel, and the editor window it opens — a big cycle display over a
grid of per-harmonic controls (phase bars, amplitude bars, frequency-scalar
spinboxes) with export/import of the wave file.

Pure UI: talks to the engine only through get_wave_params/set_wave_params.
"""

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import numpy as np
from PIL import Image as PILImage, ImageTk
from scipy.signal import stft as _stft
import matplotlib as mpl

from synthflow.gui.theme import Palette
from synthflow.core.WaveParams import WaveParams

_TWO_PI = 2.0 * math.pi


def _build_colormap_lut(name: str = "magma") -> np.ndarray:
    """256-entry RGB lookup table sampled from a real, perceptually-uniform
    matplotlib colormap (same family used by professional spectral analyzers)."""
    cmap = mpl.colormaps[name]
    return (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)


_HEAT_LUT = _build_colormap_lut()


def _draw_cycle(canvas: tk.Canvas, params: WaveParams, width: int, height: int,
                line_width: int = 1) -> None:
    canvas.delete("wave")
    samples = params.one_cycle(max(width, 2))
    peak = max(1e-9, float(abs(samples).max()))
    mid = height / 2.0
    scale = (height / 2.0 - 3) / peak
    canvas.create_line(0, mid, width, mid, fill=Palette.PANEL_EDGE, tags="wave")
    points = []
    for x in range(width):
        points.extend((x, mid - samples[x] * scale))
    canvas.create_line(*points, fill=Palette.ACCENT, width=line_width,
                       smooth=False, tags="wave")


class WaveViewer(tk.Canvas):
    """Small one-cycle display for the encoder panel; click to open the editor."""

    def __init__(self, parent, params: WaveParams, width: int = 220,
                 height: int = 56, on_click: Optional[Callable[[], None]] = None):
        super().__init__(parent, width=width, height=height, bg=Palette.INSET,
                         highlightthickness=1, highlightbackground=Palette.PANEL_EDGE,
                         cursor="hand2")
        # note: tkinter reserves self._w for the widget's Tcl pathname
        self._view_w = width
        self._view_h = height
        self.set_params(params)
        if on_click is not None:
            self.bind("<ButtonPress-1>", lambda _e: on_click())

    def set_params(self, params: WaveParams) -> None:
        _draw_cycle(self, params, self._view_w, self._view_h)


def _draw_scope(canvas: tk.Canvas, samples, f0: float, fs: float,
                width: int, height: int, line_width: int = 1) -> None:
    """Pitch-triggered scope: locks the display to a rising zero-crossing so
    the trace holds still cycle to cycle instead of scrolling/jittering."""
    canvas.delete("scope")
    mid = height / 2.0
    canvas.create_line(0, mid, width, mid, fill=Palette.PANEL_EDGE, tags="scope")
    if samples is None or len(samples) < 4 or f0 <= 0 or fs <= 0:
        return
    period = fs / f0
    span = int(min(len(samples) - 1, max(period * 3, 16)))
    search_end = max(1, len(samples) - span)
    trigger = 0
    for i in range(1, search_end):
        if samples[i - 1] <= 0.0 < samples[i]:
            trigger = i
            break
    seg = samples[trigger:trigger + span]
    if len(seg) < 2:
        return
    peak = max(1e-6, float(np.abs(seg).max()))
    scale = (height / 2.0 - 3) / peak
    n = len(seg)
    points = []
    for x in range(width):
        idx = min(n - 1, int(x / width * n))
        points.extend((x, mid - float(seg[idx]) * scale))
    canvas.create_line(*points, fill=Palette.BLUE, width=line_width, tags="scope")


class ScopeViewer(tk.Canvas):
    """Live, pitch-bound oscilloscope for the actually-playing encoded audio
    (fed samples straight from the output callback's ring buffer)."""

    def __init__(self, parent, width: int = 220, height: int = 122):
        super().__init__(parent, width=width, height=height, bg=Palette.INSET,
                         highlightthickness=1, highlightbackground=Palette.PANEL_EDGE)
        self._view_w = width
        self._view_h = height
        self.create_line(0, height / 2.0, width, height / 2.0,
                         fill=Palette.PANEL_EDGE, tags="scope")

    def update_samples(self, samples, f0: float, fs: float) -> None:
        _draw_scope(self, samples, f0, fs, self._view_w, self._view_h)


class SpectrogramViewer(tk.Canvas):
    """Scrolling log-frequency spectrogram of the live encoded audio, built
    like a DAW analyzer: a long FFT for real frequency resolution, 75%
    overlapping windows for time resolution, and the log-frequency axis
    resampled with a linear interpolation (not bucket-max) plus a Lanczos
    resize into the canvas so the display reads smooth rather than blocky.

    Speed sets how many overlapping columns are pushed per update (scroll
    rate); gain shifts the dB-to-colour mapping (brightness).
    """

    _FFT_SIZE = 4096
    _HOP = _FFT_SIZE // 4  # 75% overlap, matches typical DAW analyzers
    _BIN_COUNT = 512  # internal frequency resolution, resampled to the canvas height
    _DB_FLOOR = -90.0
    _DB_CEIL = 0.0
    _F_LO = 20.0

    def __init__(self, parent, width: int = 420, height: int = 160):
        super().__init__(parent, width=width, height=height, bg=Palette.INSET,
                         highlightthickness=1, highlightbackground=Palette.PANEL_EDGE)
        self._view_w = width
        self._view_h = height
        self._bins = np.zeros((self._BIN_COUNT, width, 3), dtype=np.uint8)
        self._bin_freqs = None
        self._freqs_fs = None
        self._image = None
        self._item = self.create_image(0, 0, anchor="nw")

    def _freqs_for(self, fs: float) -> np.ndarray:
        if self._freqs_fs != fs:
            self._bin_freqs = np.logspace(
                math.log10(self._F_LO), math.log10(fs / 2.0), self._BIN_COUNT)
            self._freqs_fs = fs
        return self._bin_freqs

    def update_samples(self, samples, fs: float, speed: float, gain_db: float) -> None:
        if samples is None or len(samples) < self._FFT_SIZE or fs <= 0:
            return
        columns = max(1, min(8, int(round(speed))))
        # scipy computes the whole overlapping-frame batch in one call, with
        # correct Hann-window normalization and no boundary padding — no
        # hand-rolled slicing/indexing to get subtly wrong.
        needed = self._FFT_SIZE + self._HOP * (columns - 1)
        tail = samples[-needed:] if len(samples) >= needed else samples
        if len(tail) < self._FFT_SIZE:
            return
        fft_freqs, _, frames = _stft(
            tail, fs=fs, window="hann", nperseg=self._FFT_SIZE,
            noverlap=self._FFT_SIZE - self._HOP, boundary=None, padded=False)
        n_frames = frames.shape[1]
        if n_frames == 0:
            return
        take = min(columns, n_frames)
        mag_db = 20.0 * np.log10(np.abs(frames[:, -take:]) + 1e-9)

        bin_freqs = self._freqs_for(fs)
        cols = []
        for i in range(take):
            db = np.interp(bin_freqs, fft_freqs, mag_db[:, i]) + gain_db
            norm = np.clip((db - self._DB_FLOOR) / (self._DB_CEIL - self._DB_FLOOR), 0.0, 1.0)
            cols.append(_HEAT_LUT[(norm * 255).astype(np.uint8)][::-1])  # low freq at bottom

        shift = len(cols)
        self._bins = np.roll(self._bins, -shift, axis=1)
        for k, col in enumerate(cols):
            self._bins[:, self._view_w - shift + k, :] = col
        image = PILImage.fromarray(self._bins, mode="RGB").resize(
            (self._view_w, self._view_h), PILImage.Resampling.LANCZOS)
        self._image = ImageTk.PhotoImage(image)
        self.itemconfigure(self._item, image=self._image)


class BarRow(tk.Canvas):
    """One row of vertical fill-bar sliders, one column per harmonic.

    Values are 0..1; drag across the row to paint. Drawn as a single canvas so
    all columns stay aligned with the spinbox row below.
    """

    def __init__(self, parent, num_bars: int, col_width: int, height: int,
                 on_change: Callable[[int, float], None],
                 fill: str = Palette.ACCENT):
        super().__init__(parent, width=num_bars * col_width, height=height,
                         bg=Palette.INSET, highlightthickness=0)
        self._n = num_bars
        self._col_w = col_width
        self._h = height
        self._fill = fill
        self._on_change = on_change
        self._values = [0.0] * num_bars
        self._rects = []
        for i in range(num_bars):
            x0 = i * col_width
            self.create_rectangle(x0, 0, x0 + col_width, height,
                                  outline=Palette.PANEL_EDGE, width=1)
            self._rects.append(self.create_rectangle(
                x0 + 2, height, x0 + col_width - 1, height,
                outline="", fill=fill))
        self.bind("<ButtonPress-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)

    def _on_drag(self, event) -> None:
        idx = event.x // self._col_w
        if not (0 <= idx < self._n):
            return
        value = min(1.0, max(0.0, 1.0 - event.y / self._h))
        self.set_value(idx, value)
        self._on_change(idx, value)

    def set_value(self, idx: int, value: float) -> None:
        self._values[idx] = value
        x0 = idx * self._col_w
        top = self._h - value * (self._h - 2)
        self.coords(self._rects[idx], x0 + 2, top, x0 + self._col_w - 1, self._h)

    def set_values(self, values) -> None:
        for i, v in enumerate(values[:self._n]):
            self.set_value(i, v)

    def get_values(self):
        return list(self._values)


class WaveEditorWindow(tk.Toplevel):
    """The full editor. Edits are pushed live through ``on_change(params)``;
    the caller forwards them to the engine and the small viewer."""

    _COL_W = 36
    _BAR_H = 72
    _WAVE_H = 200

    def __init__(self, parent, settings, params: WaveParams,
                 on_change: Callable[[WaveParams], None]):
        super().__init__(parent)
        self.title("Additive Wave Editor")
        self.configure(bg=Palette.BG, padx=12, pady=12)
        self.transient(parent)
        self._settings = settings
        self._on_change = on_change
        self._params = WaveParams(list(params.amps), list(params.phases),
                                  list(params.omegas))
        n = len(self._params.amps)

        # ── big cycle display ───────────────────────────────────────────────
        self._wave_w = max(680, min(1100, n * self._COL_W))
        self._wave = tk.Canvas(self, width=self._wave_w, height=self._WAVE_H,
                               bg=Palette.INSET, highlightthickness=1,
                               highlightbackground=Palette.PANEL_EDGE)
        self._wave.pack(fill="x")

        # ── toolbar ─────────────────────────────────────────────────────────
        bar = ttk.Frame(self, style="Bar.TFrame")
        bar.pack(fill="x", pady=(8, 8))
        ttk.Button(bar, text="Export wave…", command=self._on_export).pack(side="left")
        ttk.Button(bar, text="Import wave…", command=self._on_import).pack(
            side="left", padx=(8, 0))
        ttk.Button(bar, text="Reset to harmonic", command=self._on_reset).pack(
            side="left", padx=(8, 0))
        ttk.Label(bar, text=f"{n} harmonics", style="Bar.TLabel",
                  foreground=Palette.DIM).pack(side="right")

        # ── per-harmonic grid, horizontally scrollable ──────────────────────
        grid_w = n * self._COL_W
        view_w = self._wave_w - 70  # minus the label column
        holder = ttk.Frame(self, style="Bar.TFrame")
        holder.pack(fill="x")

        labels = ttk.Frame(holder, style="Bar.TFrame")
        labels.grid(row=0, column=0, sticky="ns")
        for r, text in ((0, "Phase"), (1, "Amp"), (2, "Freq ×")):
            ttk.Label(labels, text=text, style="Bar.TLabel", width=8,
                      foreground=Palette.DIM).grid(row=r, column=0, sticky="w",
                                                   pady=(0, 4))
            labels.rowconfigure(r, minsize=self._BAR_H + 6 if r < 2 else 28)

        self._grid_canvas = tk.Canvas(holder, width=min(grid_w, view_w),
                                      height=2 * (self._BAR_H + 6) + 34,
                                      bg=Palette.BG, highlightthickness=0)
        self._grid_canvas.grid(row=0, column=1, sticky="ew")
        holder.columnconfigure(1, weight=1)
        hsb = ttk.Scrollbar(holder, orient="horizontal",
                            command=self._grid_canvas.xview)
        hsb.grid(row=1, column=1, sticky="ew")
        self._grid_canvas.configure(xscrollcommand=hsb.set,
                                    scrollregion=(0, 0, grid_w, 0))

        inner = ttk.Frame(self._grid_canvas, style="Bar.TFrame")
        self._grid_canvas.create_window((0, 0), window=inner, anchor="nw")

        self._phase_row = BarRow(inner, n, self._COL_W, self._BAR_H,
                                 self._on_phase, fill=Palette.BLUE)
        self._phase_row.grid(row=0, column=0, pady=(0, 6))
        self._amp_row = BarRow(inner, n, self._COL_W, self._BAR_H,
                               self._on_amp, fill=Palette.ACCENT)
        self._amp_row.grid(row=1, column=0, pady=(0, 6))

        spin_row = ttk.Frame(inner, style="Bar.TFrame")
        spin_row.grid(row=2, column=0, sticky="w")
        self._omega_vars = []
        for i in range(n):
            var = tk.StringVar()
            spin = tk.Spinbox(
                spin_row, textvariable=var, width=4, from_=0.25, to=64.0,
                increment=0.25, format="%.2f", bg=Palette.INSET, fg=Palette.TEXT,
                buttonbackground=Palette.PANEL_EDGE, relief="flat",
                insertbackground=Palette.TEXT, font=Palette.FONT_SMALL,
                command=lambda i=i: self._on_omega(i))
            spin.grid(row=0, column=i, sticky="ew")
            spin_row.columnconfigure(i, minsize=self._COL_W)
            spin.bind("<Return>", lambda _e, i=i: self._on_omega(i))
            spin.bind("<FocusOut>", lambda _e, i=i: self._on_omega(i))
            self._omega_vars.append(var)

        self._load_into_widgets()
        self._redraw()

    # ── model → widgets ─────────────────────────────────────────────────────
    def _load_into_widgets(self) -> None:
        p = self._params
        self._phase_row.set_values([(ph % _TWO_PI) / _TWO_PI for ph in p.phases])
        self._amp_row.set_values([min(1.0, a) for a in p.amps])
        for var, w in zip(self._omega_vars, p.omegas):
            var.set(f"{w:.2f}")

    def _redraw(self) -> None:
        _draw_cycle(self._wave, self._params, self._wave_w, self._WAVE_H,
                    line_width=2)

    def _push(self) -> None:
        self._redraw()
        self._on_change(WaveParams(list(self._params.amps),
                                   list(self._params.phases),
                                   list(self._params.omegas)))

    # ── widget handlers ─────────────────────────────────────────────────────
    def _on_phase(self, idx: int, value: float) -> None:
        self._params.phases[idx] = value * _TWO_PI
        self._push()

    def _on_amp(self, idx: int, value: float) -> None:
        self._params.amps[idx] = value
        self._push()

    def _on_omega(self, idx: int) -> None:
        try:
            value = float(self._omega_vars[idx].get())
        except ValueError:
            self._omega_vars[idx].set(f"{self._params.omegas[idx]:.2f}")
            return
        value = min(64.0, max(0.25, value))
        self._omega_vars[idx].set(f"{value:.2f}")
        if value != self._params.omegas[idx]:
            self._params.omegas[idx] = value
            self._push()

    # ── toolbar actions ─────────────────────────────────────────────────────
    def _on_export(self) -> None:
        file_path = filedialog.asksaveasfilename(
            parent=self, title="Export wave settings", defaultextension=".json",
            filetypes=[("SSF wave files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            self._params.to_json_file(file_path)
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc), parent=self)

    def _on_import(self) -> None:
        file_path = filedialog.askopenfilename(
            parent=self, title="Import wave settings",
            filetypes=[("SSF wave files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            params = WaveParams.from_json_file(file_path)
            if len(params.amps) != self._settings.total_harmonics:
                raise ValueError(
                    f"File carries {len(params.amps)} harmonics, this rack "
                    f"runs {self._settings.total_harmonics}")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc), parent=self)
            return
        self._params = params
        self._load_into_widgets()
        self._push()

    def _on_reset(self) -> None:
        self._params = WaveParams.harmonic_default(self._settings)
        self._load_into_widgets()
        self._push()
