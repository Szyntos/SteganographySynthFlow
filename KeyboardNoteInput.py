from typing import Dict, Set

from KeyMap import KEY_TO_MIDI
from NoteState import NoteState


class KeyboardNoteInput:
    """Feeds NoteState.note_on/note_off from Tk keyboard events using the
    QWERTY-piano mapping in KeyMap, mirroring aoa_cpp_2's
    KeyboardInputWin (press/release edge detection over the same key map).

    Binds application-wide (bind_all) so held state tracks regardless of
    which widget has focus. OS key-autorepeat sends a KeyRelease
    immediately before each repeated KeyPress; a short debounce on release
    collapses that into a single continuous note_on for as long as the key
    is physically held.
    """

    _DEBOUNCE_MS = 30

    def __init__(self, widget, note_state: NoteState):
        self._widget = widget
        self._note_state = note_state
        self._pressed: Set[str] = set()
        self._pending_release: Dict[str, str] = {}
        self._enabled = False

    def enable(self) -> None:
        if self._enabled:
            return
        self._widget.bind_all("<KeyPress>", self._on_key_press)
        self._widget.bind_all("<KeyRelease>", self._on_key_release)
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        self._widget.unbind_all("<KeyPress>")
        self._widget.unbind_all("<KeyRelease>")
        self._enabled = False
        for keysym in list(self._pending_release.keys()):
            self._widget.after_cancel(self._pending_release.pop(keysym))
        for keysym in list(self._pressed):
            note = KEY_TO_MIDI.get(keysym)
            if note is not None:
                self._note_state.note_off(note)
        self._pressed.clear()

    def _on_key_press(self, event) -> None:
        keysym = event.keysym.lower()
        note = KEY_TO_MIDI.get(keysym)
        if note is None:
            return
        pending = self._pending_release.pop(keysym, None)
        if pending is not None:
            self._widget.after_cancel(pending)
        if keysym not in self._pressed:
            self._pressed.add(keysym)
            self._note_state.note_on(note)

    def _on_key_release(self, event) -> None:
        keysym = event.keysym.lower()
        note = KEY_TO_MIDI.get(keysym)
        if note is None:
            return

        def _do_release() -> None:
            self._pending_release.pop(keysym, None)
            self._pressed.discard(keysym)
            self._note_state.note_off(note)

        after_id = self._widget.after(self._DEBOUNCE_MS, _do_release)
        self._pending_release[keysym] = after_id
