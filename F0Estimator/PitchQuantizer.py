import math


def quantize_to_chromatic_hz(f_hz: float, a4_hz: float = 440.0) -> float:
    """Snap f_hz to the nearest 12-TET chromatic pitch relative to a4_hz."""
    if not math.isfinite(f_hz) or f_hz <= 0.0 or a4_hz <= 0.0:
        return 0.0
    midi = 69.0 + 12.0 * (math.log(f_hz / a4_hz) / math.log(2.0))
    midi_q = round(midi)
    f_q = a4_hz * (2.0 ** ((midi_q - 69.0) / 12.0))
    return f_q if math.isfinite(f_q) else 0.0
