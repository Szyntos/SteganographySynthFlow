"""Reusable, theme-aware building blocks for the SSF rack GUI.

Pure UI: nothing in here touches audio or DSP state.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from gui.theme import Palette

_WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}


class Panel(ttk.Frame):
    """A rack module: header strip with title + optional close button, body below."""

    def __init__(self, parent, title: str, on_close: Optional[Callable[[], None]] = None):
        super().__init__(parent, style="TFrame", padding=1)
        self.configure(borderwidth=1, relief="solid")

        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text=title, style="Title.TLabel").pack(
            side="left", padx=10, pady=4
        )
        self._header_extra = ttk.Frame(header, style="Header.TFrame")
        self._header_extra.pack(side="left", padx=8)
        if on_close is not None:
            ttk.Button(header, text="✕", style="Header.TButton", width=3,
                       command=on_close).pack(side="right", padx=4)

        self.body = ttk.Frame(self, padding=(12, 8, 12, 12))
        self.body.pack(fill="both", expand=True)

    @property
    def header_extra(self) -> ttk.Frame:
        """Slot in the header strip for small status widgets."""
        return self._header_extra


class Section(ttk.Frame):
    """Titled group inside a panel body: small dim caps label + content frame."""

    def __init__(self, parent, title: str):
        super().__init__(parent)
        ttk.Label(self, text=title.upper(), style="Section.TLabel").pack(
            anchor="w", pady=(6, 2)
        )
        self.content = ttk.Frame(self)
        self.content.pack(fill="x")


class Segmented(ttk.Frame):
    """Row of mutually exclusive toggle buttons (radio group without dots)."""

    def __init__(self, parent, options: Sequence[Tuple[str, str]],
                 command: Callable[[str], None], initial: str):
        super().__init__(parent)
        self._var = tk.StringVar(value=initial)
        self._command = command
        self._buttons: List[ttk.Radiobutton] = []
        for i, (label, value) in enumerate(options):
            btn = ttk.Radiobutton(
                self, text=label, value=value, variable=self._var,
                style="Seg.Toolbutton", command=self._fire,
            )
            btn.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 1, 0))
            self.columnconfigure(i, weight=1)
            self._buttons.append(btn)

    def _fire(self) -> None:
        self._command(self._var.get())

    def get(self) -> str:
        return self._var.get()

    def set_silent(self, value: str) -> None:
        """Move the selection without firing the command."""
        self._var.set(value)

    def set_enabled(self, enabled: bool) -> None:
        state = "!disabled" if enabled else "disabled"
        for btn in self._buttons:
            btn.state([state])


class LabeledScale(ttk.Frame):
    """Slider with a name on the left and a live value readout on the right."""

    def __init__(self, parent, text: str, from_: float, to: float,
                 fmt: Callable[[float], str],
                 command: Optional[Callable[[float], None]] = None,
                 init: Optional[float] = None, length: int = 200,
                 on_release: Optional[Callable[[float], None]] = None,
                 step: Optional[float] = None):
        super().__init__(parent)
        self._fmt = fmt
        self._command = command
        self._step = step
        self._silent = False

        ttk.Label(self, text=text, width=9, style="Dim.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self._scale = ttk.Scale(self, from_=from_, to=to, orient="horizontal",
                                length=length, command=self._on_move)
        self._scale.grid(row=0, column=1, sticky="ew", padx=(4, 6))
        self.columnconfigure(1, weight=1)
        self._value_label = ttk.Label(self, text="", width=9, anchor="e",
                                      style="Value.TLabel")
        self._value_label.grid(row=0, column=2, sticky="e")

        if on_release is not None:
            self._scale.bind("<ButtonRelease-1>",
                             lambda _e: on_release(float(self._scale.get())))
        if init is not None:
            self.set_silent(init)

    def _on_move(self, value: str) -> None:
        v = float(value)
        if self._step is not None:
            v = round(v / self._step) * self._step
        self._value_label.configure(text=self._fmt(v))
        if not self._silent and self._command is not None:
            self._command(v)

    def get(self) -> float:
        v = float(self._scale.get())
        if self._step is not None:
            v = round(v / self._step) * self._step
        return v

    def set_silent(self, value: float) -> None:
        # A disabled ttk.Scale silently ignores .set(); locked sliders still
        # have to mirror externally-driven values (note pitch, estimator,
        # follow-encoder), so lift the state around the write.
        disabled = self._scale.instate(["disabled"])
        self._silent = True
        try:
            if disabled:
                self._scale.state(["!disabled"])
            self._scale.set(value)
        finally:
            if disabled:
                self._scale.state(["disabled"])
            self._silent = False
        self._value_label.configure(text=self._fmt(float(value)))

    def set_range(self, from_: float, to: float) -> None:
        self._scale.configure(from_=from_, to=to)

    def set_enabled(self, enabled: bool) -> None:
        self._scale.state(["!disabled" if enabled else "disabled"])

    def set_readout(self, text: str) -> None:
        """Override the readout (e.g. '−∞ dB') without moving the slider."""
        self._value_label.configure(text=text)


class VBarBank(tk.Canvas):
    """Bank of vertical fill-bar sliders in the wave-editor style (BarRow):
    one column per parameter, drag to set the fill. The letter label and the
    value readout are drawn on the canvas at fixed column centres, so long
    readouts can never disturb the horizontal spacing.

    Values are normalized 0..1; the caller maps them to real units in
    ``on_change`` and formats them with ``fmt``.
    """

    _LABEL_H = 14
    _READOUT_H = 14

    def __init__(self, parent, specs: Sequence[Tuple[str, float,
                                                     Callable[[float], str],
                                                     Callable[[float], None]]],
                 col_w: int = 44, bar_h: int = 72, fill: str = Palette.ACCENT):
        """specs: (label, init_norm, fmt, on_change) per bar."""
        n = len(specs)
        height = bar_h + self._LABEL_H + self._READOUT_H
        super().__init__(parent, width=n * col_w, height=height,
                         bg=Palette.INSET, highlightthickness=0)
        self._specs = list(specs)
        self._col_w = col_w
        self._bar_h = bar_h
        self._values = [0.0] * n
        self._rects = []
        self._readouts = []
        for i, (label, init, fmt, _cmd) in enumerate(self._specs):
            x0 = i * col_w
            cx = x0 + col_w // 2
            self.create_rectangle(x0 + 4, 0, x0 + col_w - 4, bar_h,
                                  outline=Palette.PANEL_EDGE, width=1)
            self._rects.append(self.create_rectangle(
                x0 + 6, bar_h, x0 + col_w - 5, bar_h, outline="", fill=fill))
            self.create_text(cx, bar_h + self._LABEL_H // 2 + 2, text=label,
                             fill=Palette.DIM, font=Palette.FONT_SMALL)
            self._readouts.append(self.create_text(
                cx, bar_h + self._LABEL_H + self._READOUT_H // 2 + 2,
                text="", fill=Palette.TEXT, font=Palette.FONT_SMALL))
            self._set_norm(i, init)
        self.bind("<ButtonPress-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)

    def _set_norm(self, idx: int, value: float) -> None:
        value = min(1.0, max(0.0, value))
        self._values[idx] = value
        x0 = idx * self._col_w
        top = self._bar_h - value * (self._bar_h - 2)
        self.coords(self._rects[idx], x0 + 6, top, x0 + self._col_w - 5,
                    self._bar_h)
        self.itemconfigure(self._readouts[idx], text=self._specs[idx][2](value))

    def _on_drag(self, event) -> None:
        idx = event.x // self._col_w
        if not (0 <= idx < len(self._specs)):
            return
        value = min(1.0, max(0.0, 1.0 - event.y / self._bar_h))
        self._set_norm(idx, value)
        self._specs[idx][3](value)


class BipolarBar(tk.Canvas):
    """Horizontal bipolar fill bar: the fill grows from the centre to the
    left (negative) or right (positive). Value is -1..1."""

    def __init__(self, parent, init: float, fmt: Callable[[float], str],
                 on_change: Callable[[float], None],
                 width: int = 160, height: int = 18,
                 fill: str = Palette.ACCENT):
        super().__init__(parent, width=width, height=height, bg=Palette.INSET,
                         highlightthickness=0)
        self._bp_w = width
        self._bp_h = height
        self._fmt = fmt
        self._on_change = on_change
        self._readout_cb: Optional[Callable[[str], None]] = None
        self.create_rectangle(1, 1, width - 1, height - 1,
                              outline=Palette.PANEL_EDGE, width=1)
        self._rect = self.create_rectangle(width // 2, 2, width // 2,
                                           height - 2, outline="", fill=fill)
        self.create_line(width // 2, 1, width // 2, height - 1,
                         fill=Palette.DIM)
        self.set_value(init)
        self.bind("<ButtonPress-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)

    def set_readout(self, cb: Callable[[str], None]) -> None:
        self._readout_cb = cb
        cb(self._fmt(self._value))

    def set_value(self, value: float) -> None:
        self._value = min(1.0, max(-1.0, value))
        mid = self._bp_w / 2
        end = mid + self._value * (mid - 2)
        self.coords(self._rect, min(mid, end), 2, max(mid, end), self._bp_h - 2)
        if self._readout_cb is not None:
            self._readout_cb(self._fmt(self._value))

    def _on_drag(self, event) -> None:
        mid = self._bp_w / 2
        self.set_value((event.x - mid) / (mid - 2))
        self._on_change(self._value)


class FileRow(ttk.Frame):
    """Filename readout + Browse button."""

    def __init__(self, parent, initial: str, on_browse: Callable[[], None]):
        super().__init__(parent)
        self._var = tk.StringVar(value=initial)
        well = tk.Label(self, textvariable=self._var, anchor="w",
                        bg=Palette.INSET, fg=Palette.TEXT, font=Palette.FONT_MONO,
                        padx=6, pady=2)
        well.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.columnconfigure(0, weight=1)
        ttk.Button(self, text="Browse…", command=on_browse).grid(row=0, column=1)

    def set(self, name: str) -> None:
        self._var.set(name)


class Led(tk.Canvas):
    """Small round status LED."""

    def __init__(self, parent, size: int = 10, bg: str = Palette.BG):
        super().__init__(parent, width=size, height=size, bg=bg,
                         highlightthickness=0)
        self._dot = self.create_oval(1, 1, size - 1, size - 1,
                                     fill=Palette.PANEL_EDGE, outline="")

    def set_color(self, color: str) -> None:
        self.itemconfigure(self._dot, fill=color)


def make_text_well(parent, width: int = 24, height: int = 8) -> tk.Text:
    """Read-only themed text box for decoded output."""
    text = tk.Text(parent, width=width, height=height, wrap="word",
                   state="disabled", bg=Palette.INSET, fg=Palette.TEXT,
                   insertbackground=Palette.TEXT, font=Palette.FONT_MONO,
                   relief="flat", padx=6, pady=4,
                   selectbackground=Palette.ACCENT_DARK)
    return text


def set_text_well(widget: tk.Text, text: str) -> None:
    # Tcl strings cannot contain embedded NULs; raw decoded byte streams would
    # otherwise silently truncate at the first \x00.
    text = text.replace("\x00", "�")
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.see("end")
    widget.configure(state="disabled")


class ScrollFrame(ttk.Frame):
    """Scrollable container: put content in ``.inner``. Scrollbars appear only
    when the content doesn't fit; the mouse wheel scrolls vertically."""

    def __init__(self, parent, background: str = Palette.RACK):
        super().__init__(parent)
        self._canvas = tk.Canvas(self, bg=background, highlightthickness=0)
        self._vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._hsb = ttk.Scrollbar(self, orient="horizontal", command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=self._vsb.set,
                               xscrollcommand=self._hsb.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.inner = ttk.Frame(self._canvas, style="Rack.TFrame")
        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._refresh)
        self._canvas.bind("<Configure>", self._refresh)
        self._canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")

    def _refresh(self, _event=None) -> None:
        req_w = self.inner.winfo_reqwidth()
        req_h = self.inner.winfo_reqheight()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        # Stretch the content to fill the viewport when it's smaller, so
        # centred layouts (empty state) work; otherwise let it overflow.
        self._canvas.itemconfigure(self._win, width=max(req_w, cw),
                                   height=max(req_h, ch))
        self._canvas.configure(scrollregion=(0, 0, max(req_w, cw), max(req_h, ch)))
        if req_h > ch:
            self._vsb.grid(row=0, column=1, sticky="ns")
        else:
            self._vsb.grid_remove()
        if req_w > cw:
            self._hsb.grid(row=1, column=0, sticky="ew")
        else:
            self._hsb.grid_remove()

    def _on_wheel(self, event) -> None:
        if not self.winfo_exists() or not self._vsb.winfo_ismapped():
            return
        self._canvas.yview_scroll(-int(event.delta / 120), "units")


class SynthPiano(tk.Canvas):
    """Dark-themed on-screen piano.

    Highlights the active note and, when callbacks are given, is playable with
    the mouse (press = note on, release = note off).
    """

    def __init__(self, parent, low_note: int, high_note: int,
                 white_key_width: int = 26, white_key_height: int = 96,
                 on_note_on: Optional[Callable[[int], None]] = None,
                 on_note_off: Optional[Callable[[int], None]] = None):
        self._low = low_note
        self._high = high_note
        self._white_w = white_key_width
        self._white_h = white_key_height
        self._black_w = max(4, int(white_key_width * 0.6))
        self._black_h = int(white_key_height * 0.62)
        self._on_note_on = on_note_on
        self._on_note_off = on_note_off

        self._white_notes = [n for n in range(low_note, high_note + 1)
                             if n % 12 in _WHITE_PITCH_CLASSES]
        width = len(self._white_notes) * white_key_width + 2
        super().__init__(parent, width=width, height=white_key_height + 2,
                         bg=Palette.BG, highlightthickness=0)

        self._rects: Dict[int, int] = {}
        self._is_white: Dict[int, bool] = {}
        self._active_note: Optional[int] = None
        self._mouse_note: Optional[int] = None
        self._draw()

        if on_note_on is not None:
            self.bind("<ButtonPress-1>", self._on_press)
            self.bind("<B1-Motion>", self._on_drag)
            self.bind("<ButtonRelease-1>", self._on_release)

    def _white_x(self, note: int) -> int:
        return self._white_notes.index(note) * self._white_w + 1

    def _draw(self) -> None:
        for note in self._white_notes:
            x = self._white_x(note)
            rect = self.create_rectangle(
                x, 1, x + self._white_w, self._white_h,
                fill=Palette.WHITE_KEY, outline=Palette.BG, width=1,
            )
            self._rects[note] = rect
            self._is_white[note] = True

        for note in range(self._low, self._high + 1):
            if note % 12 in _WHITE_PITCH_CLASSES:
                continue
            prev_white = note - 1
            while prev_white % 12 not in _WHITE_PITCH_CLASSES:
                prev_white -= 1
            if prev_white not in self._white_notes:
                continue
            x = self._white_x(prev_white) + self._white_w - self._black_w // 2
            rect = self.create_rectangle(
                x, 1, x + self._black_w, self._black_h,
                fill=Palette.BLACK_KEY, outline=Palette.BG, width=1,
            )
            self._rects[note] = rect
            self._is_white[note] = False

    # ── mouse playing ───────────────────────────────────────────────────────
    def _note_at(self, x: int, y: int) -> Optional[int]:
        if y <= self._black_h:  # black keys sit on top
            for note, rect in self._rects.items():
                if self._is_white[note]:
                    continue
                x0, y0, x1, y1 = self.coords(rect)
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return note
        idx = (x - 1) // self._white_w
        if 0 <= idx < len(self._white_notes):
            return self._white_notes[idx]
        return None

    def _on_press(self, event) -> None:
        note = self._note_at(event.x, event.y)
        if note is not None:
            self._mouse_note = note
            self._on_note_on(note)

    def _on_drag(self, event) -> None:
        note = self._note_at(event.x, event.y)
        if note is None or note == self._mouse_note:
            return
        if self._mouse_note is not None:
            self._on_note_off(self._mouse_note)
        self._mouse_note = note
        self._on_note_on(note)

    def _on_release(self, _event) -> None:
        if self._mouse_note is not None:
            self._on_note_off(self._mouse_note)
            self._mouse_note = None

    # ── highlight ───────────────────────────────────────────────────────────
    def set_active_note(self, note: Optional[int]) -> None:
        if note == self._active_note:
            return
        if self._active_note is not None and self._active_note in self._rects:
            fill = Palette.WHITE_KEY if self._is_white[self._active_note] else Palette.BLACK_KEY
            self.itemconfigure(self._rects[self._active_note], fill=fill)
        self._active_note = note
        if note is not None and note in self._rects:
            self.itemconfigure(self._rects[note], fill=Palette.ACCENT)
