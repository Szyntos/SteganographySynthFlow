from enum import Enum
from typing import List, Optional

import numpy as np
from scipy.signal import resample_poly

from Payload.pixel_codec import AudioDigitalCodec
from Settings import Settings


class ResampleMethod(str, Enum):
    POLY = "poly"
    LINEAR = "linear"
    HOLD = "hold"


class RowToAudioResampler:
    # Symbol rows are resampled from their native (data_harmonics-sample)
    # rate up to chunk_size in polyphase batches rather than row-by-row, and
    # a few trailing rows of context are carried across batches. A per-row
    # linear stretch (the old approach) sounds noticeably worse than a WAV
    # dump of the same decoded audio because linear interpolation over a
    # ~10x upsample factor is a poor reconstruction filter; batching lets
    # resample_poly apply a proper band-limited filter instead, at the cost
    # of a little extra latency.
    #
    # resample_poly needs symmetric filter support (samples both before AND
    # after the point being reconstructed) to interpolate a sharp value
    # change without ringing. _OVERLAP_ROWS only supplies past context, so
    # the trailing edge of every batch was always under-supported and rang
    # on every row boundary — a defect recurring on a fixed schedule, heard
    # as rhythmic clicks. _LOOKAHEAD_ROWS fixes this structurally: rows are
    # only emitted once _LOOKAHEAD_ROWS further rows have already arrived,
    # so those future rows can be included as trailing context for the
    # resample (and then discarded, not emitted) — at the cost of extra
    # output latency instead of extra ringing.
    # (Batch/overlap/lookahead sizes are read from settings; see reconfigure.)

    def __init__(self, settings: Settings, resample_method: "ResampleMethod | str" = ResampleMethod.POLY):
        self._settings = settings
        self._batch_rows: int = settings.decoder_batch_rows
        self._overlap_rows: int = settings.decoder_overlap_rows
        self._lookahead_rows: int = settings.decoder_lookahead_rows
        self._chunk_size: int = 0
        self._data_harmonics: int = 0
        # When set (audio payload + DIGITAL codec), each row's raw per-harmonic
        # values are bit-packed symbols, not smooth amplitude samples — they
        # must be unpacked via decode_chunk before resampling, or the packed
        # bit-jitter aliases through the reconstruction filter instead of the
        # real audio content it encodes.
        self._audio_codec: Optional[AudioDigitalCodec] = None
        self._row_len: int = 0
        self._pending_rows: List[List[float]] = []
        self._history: List[float] = []
        self.set_resample_method(resample_method)

    def set_resample_method(self, resample_method: "ResampleMethod | str") -> None:
        self._resample_method = ResampleMethod(resample_method)

    def get_resample_method(self) -> ResampleMethod:
        return self._resample_method

    def set_audio_codec(self, audio_codec: Optional[AudioDigitalCodec]) -> None:
        self._audio_codec = audio_codec
        self._row_len = self._data_harmonics * (
            audio_codec.samples_per_symbol if audio_codec is not None else 1
        )
        self._pending_rows = []
        self._history = []

    def reconfigure(self, chunk_size: int, data_harmonics: int) -> None:
        self._chunk_size = chunk_size
        self._data_harmonics = data_harmonics
        self._row_len = data_harmonics * (
            self._audio_codec.samples_per_symbol if self._audio_codec is not None else 1
        )
        self._pending_rows = []
        self._history = []

    def push_row(self, offsets: List[float]) -> List[float]:
        if self._audio_codec is not None:
            decoded: List[float] = []
            for level in offsets:
                decoded.extend(self._audio_codec.decode_chunk([level]))
            offsets = decoded
        self._pending_rows.append(offsets)
        if len(self._pending_rows) >= self._batch_rows + self._lookahead_rows:
            return self._resample_pending_rows()
        return []

    def _resample_pending_rows(self) -> List[float]:
        data_harmonics = self._row_len
        chunk_size = self._chunk_size

        # Only the first _batch_rows rows are emitted this call; the
        # remaining _lookahead_rows rows are already-decoded future data,
        # included below purely as trailing filter context, then left in
        # _pending_rows to be re-used (as batch or lookahead) next call.
        emit_rows = self._pending_rows[:self._batch_rows]
        lookahead_rows = self._pending_rows[self._batch_rows:]
        n_rows = len(emit_rows)
        batch: List[float] = [s for row in emit_rows for s in row]
        lookahead: List[float] = [s for row in lookahead_rows for s in row]

        overlap_rows = min(self._overlap_rows, len(self._history) // data_harmonics)
        overlap_samples = overlap_rows * data_harmonics
        context = self._history[len(self._history) - overlap_samples:] if overlap_samples else []

        native = np.array(context + batch + lookahead, dtype=np.float64)
        resampled = self._upsample(native, chunk_size, data_harmonics)

        overlap_out = overlap_rows * chunk_size
        new_out = n_rows * chunk_size
        result = resampled[overlap_out:overlap_out + new_out]

        # Carry forward up to _overlap_rows rows of context for the next
        # batch. Must be drawn from (context + batch), not just batch, or
        # the history can never grow past _batch_rows rows regardless of
        # _overlap_rows.
        full_history = context + batch
        tail_rows = min(self._overlap_rows, overlap_rows + n_rows)
        self._history = full_history[len(full_history) - tail_rows * data_harmonics:]
        self._pending_rows = lookahead_rows

        return result.tolist()

    def _upsample(self, native: np.ndarray, chunk_size: int, data_harmonics: int) -> np.ndarray:
        if self._resample_method is ResampleMethod.POLY:
            return resample_poly(native, chunk_size, data_harmonics)

        out_len = (len(native) // data_harmonics) * chunk_size

        if self._resample_method is ResampleMethod.HOLD:
            # Zero-order hold: repeat each native sample chunk_size/data_harmonics
            # times. The most basic possible reconstruction — a staircase, not
            # a smooth curve — but with no filter transient to reason about.
            idx = (np.arange(out_len) * len(native)) // out_len
            return native[np.minimum(idx, len(native) - 1)]

        if self._resample_method is ResampleMethod.LINEAR:
            # Continuous linear interpolation over the whole batch+context
            # array (not per-row), so there's no seam between symbol rows
            # within a batch. Still a poor reconstruction filter for a ~20x
            # upsample (aliasing/roughness vs "poly"), but simple and has no
            # cross-batch filter-transient behaviour to get wrong.
            native_idx = np.arange(len(native), dtype=np.float64)
            out_idx = np.arange(out_len, dtype=np.float64) * (len(native) - 1) / (out_len - 1)
            return np.interp(out_idx, native_idx, native)

        raise ValueError(f"Unknown resample_method: {self._resample_method!r}")
