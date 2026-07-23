from typing import Callable


class MidiInput:
    def __init__(self):
        self._on_play_callback: Callable[[float], None] | None = None

    def on_play(self, function: Callable[[float], None]) -> None:
        self._on_play_callback = function

    def trigger(self, note: float) -> None:
        if self._on_play_callback is not None:
            self._on_play_callback(note)
