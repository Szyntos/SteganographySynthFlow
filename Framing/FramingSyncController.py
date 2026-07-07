from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

from Settings import Settings


def hamming_distance(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    distance = sum(1 for i in range(n) if (a[i] & 1) != (b[i] & 1))
    return distance + abs(len(a) - len(b))


def _slice_pattern(pattern: List[int], chunk_len: int) -> List[List[int]]:
    slices: List[List[int]] = []
    for offset in range(0, len(pattern), chunk_len):
        piece = pattern[offset:offset + chunk_len]
        if len(piece) < chunk_len:
            piece = piece + [0] * (chunk_len - len(piece))
        slices.append(piece)
    return slices


class _SyncDetector:
    """Fires True exactly once when the stream EXITs the sync region."""

    def __init__(self, pattern_slices: List[List[int]], window_len: int, min_match: int,
                 fuzzy_max_errors: int, data_diff_min_errors: int):
        self._slices = pattern_slices
        self._window_len = max(1, window_len)
        self._min_match = max(1, min_match)
        self._fuzzy_max_errors = fuzzy_max_errors
        self._data_diff_min_errors = max(1, data_diff_min_errors)
        self._window: Deque[bool] = deque()
        self._armed = False
        self._just_fired = False

    def reset(self) -> None:
        self._window.clear()
        self._armed = False
        self._just_fired = False

    def _distance(self, bits: Sequence[int]) -> int:
        return min(hamming_distance(bits, s) for s in self._slices)

    def push(self, bits: Sequence[int]) -> bool:
        dist = self._distance(bits)
        is_match = dist <= self._fuzzy_max_errors

        self._window.append(is_match)
        while len(self._window) > self._window_len:
            self._window.popleft()

        if not self._armed:
            if len(self._window) == self._window_len:
                if sum(1 for m in self._window if m) >= self._min_match:
                    self._armed = True
            return False

        if not is_match and dist >= self._data_diff_min_errors:
            if not self._just_fired:
                self._just_fired = True
                return True

        return False


class FramingSyncController:
    """Fuzzy start/end frame detector.

    Pushed one bit-chunk per symbol row; each chunk is Hamming-matched against
    the frame_start / frame_end patterns (per chunk-sized pattern slice, since
    a sync marker spans len(pattern) / chunk_len consecutive rows in the
    serialized stream). Codec-agnostic: only ever sees bit patterns.
    """

    def __init__(self,
                 start_bits: Optional[List[int]] = None,
                 end_bits: Optional[List[int]] = None,
                 chunk_len: Optional[int] = None,
                 window_len: int = 1,
                 min_match: int = 1,
                 fuzzy_max_errors: int = 1,
                 data_diff_min_errors: int = 1):
        self._enabled = bool(start_bits) and bool(end_bits)
        if not self._enabled:
            return

        if start_bits == end_bits:
            raise ValueError("FramingSyncController: start/end patterns must differ")

        chunk_len = chunk_len if chunk_len else max(len(start_bits), len(end_bits))
        self._window_len = max(1, window_len)
        self._min_match = max(1, min(min_match, self._window_len))
        self._fuzzy_max_errors = fuzzy_max_errors

        self._end_slices = _slice_pattern(end_bits, chunk_len)
        self._start_detector = _SyncDetector(
            _slice_pattern(start_bits, chunk_len),
            self._window_len, self._min_match, fuzzy_max_errors, data_diff_min_errors,
        )
        self._end_window: Deque[bool] = deque()

    @classmethod
    def from_settings(cls, settings: Settings) -> "FramingSyncController":
        from Framing.FrameGenerator import FrameGenerator

        frame_generator = FrameGenerator(settings)
        start_bits = frame_generator.get_start()
        end_bits = frame_generator.get_end()

        chunk_len = settings.data_harmonics
        # A sync marker spans ceil(len(pattern) / chunk_len) rows in the stream
        # (load_payload inserts it exactly once), so the effective window can
        # never usefully exceed that span.
        span = max(1, -(-len(start_bits) // chunk_len))
        window_len = max(1, min(settings.sync_window, span))
        min_match = max(1, min(settings.sync_min_match, window_len))
        fuzzy_max_errors = max(1, round(settings.sync_fuzzy_max_bit_errors_frac * chunk_len))
        data_diff_min_errors = max(1, round(settings.sync_data_diff_frac * chunk_len))

        return cls(start_bits, end_bits, chunk_len,
                   window_len, min_match, fuzzy_max_errors, data_diff_min_errors)

    @staticmethod
    def quantize_row_to_bits(offsets: Sequence[float]) -> List[int]:
        return [1 if value >= 0.5 else 0 for value in offsets]

    def reset(self) -> None:
        if not self._enabled:
            return
        self._start_detector.reset()
        self._end_window.clear()

    def push(self, bits: Sequence[int]) -> Tuple[bool, bool, bool]:
        """Returns (start_fire, end_fire_enter, is_end_match_now)."""
        if not self._enabled:
            return False, False, False

        start_fire = self._start_detector.push(bits)

        dist_end = min(hamming_distance(bits, s) for s in self._end_slices)
        is_end_match = dist_end <= self._fuzzy_max_errors

        self._end_window.append(is_end_match)
        while len(self._end_window) > self._window_len:
            self._end_window.popleft()

        end_fire_enter = False
        if len(self._end_window) == self._window_len:
            count = sum(1 for m in self._end_window if m)
            end_fire_enter = count >= self._min_match

        return start_fire, end_fire_enter, is_end_match
