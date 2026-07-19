import threading
import time
from typing import Optional

from NoteState import NoteState

try:
    import mido
except ImportError:
    mido = None


class MidiFilePlayer:
    """Plays back a standard MIDI file into a NoteState on a background thread.

    Feeding the same NoteState the live MIDI/keyboard inputs use means the
    encoder gating and the on-screen piano highlight work identically for
    file playback. The tempo scale is read per message, so moving the tempo
    knob mid-playback takes effect immediately.
    """

    def __init__(self, note_state: NoteState):
        self._note_state = note_state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._tempo_scale: float = 1.0
        self._loop: bool = False
        self._transpose: int = 0
        # Maps an original note to the transposed note it was sounded as, so a
        # note-off still releases the right key if the transpose knob moved
        # while the note was held.
        self._sounding: dict = {}
        self._file_path: Optional[str] = None
        self._midi = None
        self._playing = False

    def load(self, file_path: str) -> None:
        if mido is None:
            raise RuntimeError("mido is not installed.\nRun: pip install mido python-rtmidi")
        self._midi = mido.MidiFile(file_path)
        self._file_path = file_path

    def get_file_path(self) -> Optional[str]:
        return self._file_path

    def set_tempo_scale(self, scale: float) -> None:
        self._tempo_scale = max(0.05, float(scale))

    def get_tempo_scale(self) -> float:
        return self._tempo_scale

    def set_loop(self, loop: bool) -> None:
        self._loop = bool(loop)

    def get_loop(self) -> bool:
        return self._loop

    def set_transpose(self, semitones: int) -> None:
        self._transpose = int(semitones)

    def get_transpose(self) -> int:
        return self._transpose

    def is_playing(self) -> bool:
        return self._playing

    def start(self) -> None:
        if self._midi is None:
            raise RuntimeError("No MIDI file loaded.")
        self.stop()
        self._stop_event.clear()
        self._playing = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self._playing = False
        self._note_state.reset()

    def _run(self) -> None:
        try:
            while not self._play_once():
                if not self._loop:
                    return
                self._note_state.reset()
                self._sounding.clear()
        finally:
            self._playing = False
            self._note_state.reset()
            self._sounding.clear()

    def _play_once(self) -> bool:
        """Play the file through once. Returns True if stopped mid-file."""
        # Iterating a MidiFile yields messages with .time already converted
        # to seconds (embedded tempo changes included), so scaling the
        # inter-message delay is all a tempo knob needs to do.
        for msg in self._midi:
            deadline = time.monotonic() + msg.time / self._tempo_scale
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if self._stop_event.wait(min(0.05, remaining)):
                    return True
            if msg.type == "note_on" and msg.velocity > 0:
                note = msg.note + self._transpose
                if 0 <= note <= 127:
                    self._sounding[msg.note] = note
                    self._note_state.note_on(note)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                note = self._sounding.pop(msg.note, None)
                if note is not None:
                    self._note_state.note_off(note)
        return self._stop_event.is_set()
