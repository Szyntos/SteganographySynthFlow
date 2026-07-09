from typing import List, Optional

from NoteState import NoteState

try:
    import mido
except ImportError:
    mido = None


def list_midi_input_devices() -> List[str]:
    if mido is None:
        return []
    try:
        return list(mido.get_input_names())
    except Exception:
        return []


class MidiDeviceInput:
    """Feeds real-time note-on/note-off events from a MIDI input port into a NoteState.

    Uses mido's callback-based port, so events arrive on mido/rtmidi's own
    background thread; NoteState is internally lock-protected so this is safe
    to read from the audio callback thread concurrently.
    """

    def __init__(self, note_state: NoteState):
        self._note_state = note_state
        self._port = None

    def is_running(self) -> bool:
        return self._port is not None

    def start(self, device_name: Optional[str] = None) -> None:
        if mido is None:
            raise RuntimeError("mido is not installed.\nRun: pip install mido python-rtmidi")
        self.stop()
        self._port = (
            mido.open_input(device_name, callback=self._handle)
            if device_name else mido.open_input(callback=self._handle)
        )

    def stop(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None
        self._note_state.reset()

    def _handle(self, msg) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            self._note_state.note_on(msg.note)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            self._note_state.note_off(msg.note)
