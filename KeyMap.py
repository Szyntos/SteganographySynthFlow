"""QWERTY-piano key -> MIDI note mapping, ported from aoa_cpp_2's keymap.hpp.

Keys are Tk keysyms (lowercased). C4 = MIDI note 60. The upper row is a
deliberate overlapping continuation of the lower row (both cover C5-E5),
exactly mirroring the C++ layout.
"""

_C4 = 60

KEY_TO_MIDI = {
    # lower row: Z S X D C V G B H N J M , L . ; /
    "z": _C4 + 0,
    "s": _C4 + 1,
    "x": _C4 + 2,
    "d": _C4 + 3,
    "c": _C4 + 4,
    "v": _C4 + 5,
    "g": _C4 + 6,
    "b": _C4 + 7,
    "h": _C4 + 8,
    "n": _C4 + 9,
    "j": _C4 + 10,
    "m": _C4 + 11,
    "comma": _C4 + 12,
    "l": _C4 + 13,
    "period": _C4 + 14,
    "semicolon": _C4 + 15,
    "slash": _C4 + 16,

    # upper row: Q 2 W 3 E R 5 T 6 Y 7 U I 9 O 0 P
    "q": _C4 + 12,
    "2": _C4 + 13,
    "w": _C4 + 14,
    "3": _C4 + 15,
    "e": _C4 + 16,
    "r": _C4 + 17,
    "5": _C4 + 18,
    "t": _C4 + 19,
    "6": _C4 + 20,
    "y": _C4 + 21,
    "7": _C4 + 22,
    "u": _C4 + 23,
    "i": _C4 + 24,
    "9": _C4 + 25,
    "o": _C4 + 26,
    "0": _C4 + 27,
    "p": _C4 + 28,
}
