import threading
from typing import List


def midi_note_to_hz(midi_note: int, a4_hz: float = 440.0) -> float:
    return a4_hz * (2.0 ** ((midi_note - 69) / 12.0))


class NoteState:
    """Monophonic, last-note-priority held-note stack.

    Ported from aoa_cpp_2's NoteState: the most recently pressed held note
    wins. Releasing the active note reveals whichever note is still held
    underneath, instead of jumping to silence or an arbitrary other note.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._held: List[int] = []

    def note_on(self, midi_note: int) -> None:
        with self._lock:
            if midi_note in self._held:
                self._held.remove(midi_note)
            self._held.append(midi_note)

    def note_off(self, midi_note: int) -> None:
        with self._lock:
            if midi_note in self._held:
                self._held.remove(midi_note)

    def has_note(self) -> bool:
        with self._lock:
            return len(self._held) > 0

    def current_note_or(self, fallback: int) -> int:
        with self._lock:
            if not self._held:
                return fallback
            return self._held[-1]

    def reset(self) -> None:
        with self._lock:
            self._held.clear()
