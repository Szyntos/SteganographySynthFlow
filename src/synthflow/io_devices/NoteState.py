import threading
from typing import List, Tuple


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
        # (midi_note, generation) pairs; generation is a monotonic counter
        # bumped on every note_on so a fast off/on of the same pitch between
        # two polls of the audio thread still looks like a change to
        # current_gen_or(), even though current_note_or() alone would not
        # have moved.
        self._held: List[Tuple[int, int]] = []
        self._gen = 0

    def note_on(self, midi_note: int) -> None:
        with self._lock:
            self._held = [pair for pair in self._held if pair[0] != midi_note]
            self._gen += 1
            self._held.append((midi_note, self._gen))

    def note_off(self, midi_note: int) -> None:
        with self._lock:
            self._held = [pair for pair in self._held if pair[0] != midi_note]

    def has_note(self) -> bool:
        with self._lock:
            return len(self._held) > 0

    def current_note_or(self, fallback: int) -> int:
        with self._lock:
            if not self._held:
                return fallback
            return self._held[-1][0]

    def current_gen_or(self, fallback: int) -> int:
        """Generation number of the currently sounding note, so callers can
        detect a same-pitch note-off/note-on that happened between polls."""
        with self._lock:
            if not self._held:
                return fallback
            return self._held[-1][1]

    def reset(self) -> None:
        with self._lock:
            self._held.clear()
