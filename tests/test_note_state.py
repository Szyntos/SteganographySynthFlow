import math

from synthflow.io_devices.NoteState import NoteState, midi_note_to_hz


def test_midi_note_to_hz_a4_and_octave():
    assert midi_note_to_hz(69) == 440.0
    assert math.isclose(midi_note_to_hz(81), 880.0, rel_tol=1e-9)


def test_last_note_priority_reveals_underlying_note():
    ns = NoteState()
    ns.note_on(60)
    ns.note_on(64)
    assert ns.current_note_or(-1) == 64
    ns.note_off(64)
    assert ns.current_note_or(-1) == 60


def test_repress_moves_note_to_top():
    ns = NoteState()
    ns.note_on(60)
    ns.note_on(64)
    ns.note_on(60)
    assert ns.current_note_or(-1) == 60
    ns.note_off(60)
    assert ns.current_note_or(-1) == 64


def test_current_note_or_fallback_when_empty():
    ns = NoteState()
    assert ns.current_note_or(42) == 42
    assert ns.has_note() is False


def test_reset_clears_held_notes():
    ns = NoteState()
    ns.note_on(60)
    ns.reset()
    assert ns.has_note() is False
    assert ns.current_note_or(-1) == -1
