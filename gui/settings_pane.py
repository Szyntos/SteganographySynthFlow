"""Hideable 'Advanced DSP' pane: live-editable transmission geometry.

Pure UI: the pane only parses entries and hands a {setting-name: value} dict
to its on_apply callback; all DSP rebuilding happens behind the engines.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict

from gui.theme import Palette
from Settings import Settings

# (label, flat settings attribute, parser)
_FIELDS = (
    ("Chunk size", "base_chunk_size", int),
    ("Total harm", "total_harmonics", int),
    ("Data harm", "data_harmonics", int),
    ("Data offset", "data_offset", int),
    ("Phase range", "phase_range", float),
    ("Image W", "image_target_w", int),
    ("Image H", "image_target_h", int),
    ("Img chans", "image_channels", int),
)


class DspSettingsPane(ttk.Frame):
    """Collapsed by default; the header button toggles the field grid.

    on_apply receives the parsed values and should raise if they are
    rejected; the entries are refreshed from the live settings afterwards
    either way, so a failed apply visibly snaps back.
    """

    def __init__(self, parent, settings: Settings,
                 on_apply: Callable[[Dict], None]):
        super().__init__(parent)
        self._settings = settings
        self._on_apply = on_apply
        self._open = False

        self._toggle_btn = ttk.Button(self, text="▸ Advanced DSP",
                                      style="Header.TButton", command=self._toggle)
        self._toggle_btn.pack(anchor="w", pady=(8, 0))

        self._content = ttk.Frame(self)
        self._vars: Dict[str, tk.StringVar] = {}
        for i, (label, attr, _parse) in enumerate(_FIELDS):
            r, c = divmod(i, 2)
            ttk.Label(self._content, text=label, style="Dim.TLabel", width=11).grid(
                row=r, column=c * 2, sticky="w", pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(self._content, textvariable=var, width=8,
                              font=Palette.FONT_MONO)
            entry.grid(row=r, column=c * 2 + 1, sticky="w",
                       padx=(0, 12 if c == 0 else 0), pady=2)
            entry.bind("<Return>", lambda _e: self._apply())
            self._vars[attr] = var

        rows = (len(_FIELDS) + 1) // 2
        ttk.Button(self._content, text="Apply", style="Accent.TButton",
                   command=self._apply).grid(row=rows, column=0, columnspan=4,
                                             sticky="w", pady=(6, 0))
        self.refresh()

    def _toggle(self) -> None:
        self._open = not self._open
        if self._open:
            self.refresh()
            self._content.pack(fill="x", pady=(4, 0))
        else:
            self._content.pack_forget()
        self._toggle_btn.configure(
            text=("▾" if self._open else "▸") + " Advanced DSP")

    def refresh(self) -> None:
        """Mirror the live settings into the entries."""
        for _label, attr, parse in _FIELDS:
            value = getattr(self._settings, attr)
            self._vars[attr].set(f"{value:.6g}" if parse is float else str(value))

    def _parse(self) -> Dict:
        values = {}
        for label, attr, parse in _FIELDS:
            text = self._vars[attr].get().strip()
            try:
                values[attr] = parse(text)
            except ValueError:
                raise ValueError(f"{label}: '{text}' is not a valid "
                                 f"{'number' if parse is float else 'integer'}")
        return values

    def _apply(self) -> None:
        try:
            self._on_apply(self._parse())
        except Exception as exc:
            messagebox.showerror("Advanced DSP Error", str(exc))
        finally:
            self.refresh()
