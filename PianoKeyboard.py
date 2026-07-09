import tkinter as tk
from typing import Dict, Optional

_WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}


class PianoKeyboard(tk.Canvas):
    """Minimal on-screen piano that highlights the currently held note."""

    def __init__(
        self,
        master,
        low_note: int = 60,
        high_note: int = 88,
        white_key_width: int = 22,
        white_key_height: int = 90,
        **kwargs,
    ):
        self._low = low_note
        self._high = high_note
        self._white_w = white_key_width
        self._white_h = white_key_height
        self._black_w = max(4, int(white_key_width * 0.6))
        self._black_h = int(white_key_height * 0.6)

        self._white_notes = [n for n in range(low_note, high_note + 1) if n % 12 in _WHITE_PITCH_CLASSES]
        width = len(self._white_notes) * white_key_width + 2
        height = white_key_height + 2

        super().__init__(master, width=width, height=height, highlightthickness=0, **kwargs)

        self._rects: Dict[int, int] = {}
        self._is_white: Dict[int, bool] = {}
        self._active_note: Optional[int] = None

        self._draw()

    def _white_x(self, note: int) -> int:
        return self._white_notes.index(note) * self._white_w

    def _draw(self) -> None:
        for note in self._white_notes:
            x = self._white_x(note)
            rect = self.create_rectangle(
                x, 0, x + self._white_w, self._white_h, fill="white", outline="black",
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
                x, 0, x + self._black_w, self._black_h, fill="black", outline="black",
            )
            self._rects[note] = rect
            self._is_white[note] = False

    def set_active_note(self, note: Optional[int]) -> None:
        if note == self._active_note:
            return
        if self._active_note is not None and self._active_note in self._rects:
            self._restore_color(self._active_note)
        self._active_note = note
        if note is not None and note in self._rects:
            self.itemconfigure(self._rects[note], fill="#3aa0ff")

    def _restore_color(self, note: int) -> None:
        color = "white" if self._is_white.get(note, True) else "black"
        self.itemconfigure(self._rects[note], fill=color)
