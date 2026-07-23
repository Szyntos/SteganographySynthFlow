"""Dark synth-rack theme for the SSF GUI.

All colours and ttk style names live here so panels/widgets never hardcode
colours. Apply once per Tk root via ``apply_theme(root)``.
"""

from tkinter import ttk


class Palette:
    BG = "#141519"          # window background
    RACK = "#101114"        # rack well behind modules
    PANEL = "#1e2027"       # module face
    PANEL_EDGE = "#2c2f3a"  # module border / separators
    INSET = "#15161b"       # slider troughs, text wells
    TEXT = "#d8dbe4"
    DIM = "#8a8f9e"
    ACCENT = "#e8a24c"      # amber — active / selected
    ACCENT_DARK = "#7a5426"
    BLUE = "#5db2ff"        # value readouts
    GREEN = "#48d17a"       # running LED
    RED = "#e05a5a"         # warnings / weak signal
    WHITE_KEY = "#dcdee6"
    BLACK_KEY = "#22242c"

    FONT = ("Segoe UI", 9)
    FONT_SMALL = ("Segoe UI", 8)
    FONT_BOLD = ("Segoe UI Semibold", 9)
    FONT_TITLE = ("Segoe UI Semibold", 10)
    FONT_MONO = ("Consolas", 9)


def apply_theme(root) -> None:
    root.configure(bg=Palette.RACK)
    style = ttk.Style(root)
    style.theme_use("clam")
    P = Palette

    style.configure(".", background=P.PANEL, foreground=P.TEXT, font=P.FONT,
                    bordercolor=P.PANEL_EDGE, darkcolor=P.PANEL, lightcolor=P.PANEL,
                    troughcolor=P.INSET, fieldbackground=P.INSET,
                    selectbackground=P.ACCENT_DARK, selectforeground=P.TEXT,
                    insertcolor=P.TEXT)

    style.configure("TFrame", background=P.PANEL)
    style.configure("Rack.TFrame", background=P.RACK)
    style.configure("Bar.TFrame", background=P.BG)
    style.configure("Header.TFrame", background=P.PANEL_EDGE)

    style.configure("TLabel", background=P.PANEL, foreground=P.TEXT)
    style.configure("Bar.TLabel", background=P.BG, foreground=P.TEXT)
    style.configure("Title.TLabel", background=P.PANEL_EDGE, foreground=P.ACCENT,
                    font=P.FONT_TITLE)
    style.configure("Section.TLabel", background=P.PANEL, foreground=P.DIM,
                    font=P.FONT_SMALL)
    style.configure("Value.TLabel", background=P.PANEL, foreground=P.BLUE,
                    font=P.FONT_MONO)
    style.configure("Dim.TLabel", background=P.PANEL, foreground=P.DIM)
    style.configure("Warn.TLabel", background=P.BG, foreground=P.RED,
                    font=P.FONT_BOLD)

    style.configure("TButton", background=P.PANEL_EDGE, foreground=P.TEXT,
                    borderwidth=1, focusthickness=0, padding=(10, 3))
    style.map("TButton",
              background=[("active", P.ACCENT_DARK), ("pressed", P.ACCENT_DARK)])
    style.configure("Accent.TButton", background=P.ACCENT, foreground="#1a1206",
                    font=P.FONT_BOLD, padding=(14, 4))
    style.map("Accent.TButton",
              background=[("active", "#f4b96b"), ("pressed", "#c9883c"),
                          ("disabled", P.PANEL_EDGE)],
              foreground=[("disabled", P.DIM)])
    style.configure("Header.TButton", background=P.PANEL_EDGE, foreground=P.DIM,
                    padding=(4, 0), borderwidth=0)
    style.map("Header.TButton", foreground=[("active", P.RED)],
              background=[("active", P.PANEL_EDGE)])

    # Segmented radio groups (Toolbutton = no indicator dot).
    style.configure("Seg.Toolbutton", background=P.INSET, foreground=P.DIM,
                    padding=(10, 3), borderwidth=1, anchor="center")
    style.map("Seg.Toolbutton",
              background=[("disabled", P.PANEL), ("selected", P.ACCENT),
                          ("active", P.PANEL_EDGE)],
              foreground=[("disabled", P.PANEL_EDGE), ("selected", "#1a1206"),
                          ("active", P.TEXT)])

    style.configure("TCheckbutton", background=P.PANEL, foreground=P.TEXT)
    style.map("TCheckbutton",
              background=[("active", P.PANEL)],
              foreground=[("disabled", P.DIM)],
              indicatorcolor=[("selected", P.ACCENT), ("!selected", P.INSET)])

    style.configure("Horizontal.TScale", troughcolor=P.INSET, background=P.ACCENT,
                    bordercolor=P.PANEL_EDGE, lightcolor=P.ACCENT, darkcolor=P.ACCENT,
                    gripcount=0)
    style.map("Horizontal.TScale",
              background=[("disabled", P.PANEL_EDGE)],
              troughcolor=[("disabled", P.PANEL)])

    style.configure("TCombobox", arrowcolor=P.DIM, padding=(6, 2))
    style.map("TCombobox",
              fieldbackground=[("readonly", P.INSET), ("disabled", P.PANEL)],
              foreground=[("disabled", P.DIM)],
              selectbackground=[("readonly", P.INSET)],
              selectforeground=[("readonly", P.TEXT)])
    root.option_add("*TCombobox*Listbox.background", P.INSET)
    root.option_add("*TCombobox*Listbox.foreground", P.TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", P.ACCENT_DARK)
    root.option_add("*TCombobox*Listbox.selectForeground", P.TEXT)

    style.configure("TSpinbox", arrowcolor=P.DIM, padding=(4, 1))
    style.map("TSpinbox", fieldbackground=[("disabled", P.PANEL)])

    style.configure("Vertical.TScrollbar", background=P.PANEL_EDGE,
                    troughcolor=P.INSET, arrowcolor=P.DIM)
