from typing import List

import numpy as np
from scipy.signal import resample_poly

from AudioChunk import AudioChunk
from Payload import SymbolRow
from SamplesFifo import SamplesFifo
from Settings import Settings
from Sink import SymbolSink
from .DecodingStrategy import DecodingStrategy


RESAMPLE_METHODS: List[str] = ["poly", "linear", "hold"]


class Decoder:
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
    _BATCH_ROWS: int = 4
    _OVERLAP_ROWS: int = 20
    _LOOKAHEAD_ROWS: int = 4

    def __init__(
            self,
            settings: Settings,
            decoding_strategy: DecodingStrategy,
            sink: SymbolSink,
            resample_method: str = "poly",
    ):
        self._settings = settings
        self._decoding_strategy: DecodingStrategy = decoding_strategy
        self._sink: SymbolSink = sink
        self._max_driver_block_size: int = 0
        self._audio_chunk_output_fifo: SamplesFifo = SamplesFifo()
        self._pending_rows: List[List[float]] = []
        self._history: List[float] = []
        self.set_resample_method(resample_method)
        self.reconfigure()

    def set_resample_method(self, resample_method: str) -> None:
        if resample_method not in RESAMPLE_METHODS:
            raise ValueError(f"Unknown resample_method: {resample_method!r}")
        self._resample_method = resample_method

    def get_resample_method(self) -> str:
        return self._resample_method

    def reconfigure(self) -> None:
        self._max_driver_block_size = self._settings.max_driver_block_size
        self._audio_chunk_output_fifo = SamplesFifo(
            self._decoding_strategy.get_internal_clock() + self._max_driver_block_size - 1
        )
        self._pending_rows = []
        self._history = []

    def get_output_fifo_size(self) -> int:
        return self._audio_chunk_output_fifo.get_size()

    def process(self, input_samples: AudioChunk, num_samples: int) -> AudioChunk:
        decoded_symbols: List[SymbolRow] = self._decoding_strategy.decode_samples(input_samples, num_samples)
        for symbol_row in decoded_symbols:
            self._pending_rows.append(symbol_row.get_offsets())
            if len(self._pending_rows) >= self._BATCH_ROWS + self._LOOKAHEAD_ROWS:
                self._resample_pending_rows()
        self._sink.push_many(decoded_symbols)
        return AudioChunk(self._audio_chunk_output_fifo.pop_or_silence(num_samples))

    def _resample_pending_rows(self) -> None:
        data_harmonics = self._settings.data_harmonics
        chunk_size = self._decoding_strategy.get_internal_clock()

        # Only the first _BATCH_ROWS rows are emitted this call; the
        # remaining _LOOKAHEAD_ROWS rows are already-decoded future data,
        # included below purely as trailing filter context, then left in
        # _pending_rows to be re-used (as batch or lookahead) next call.
        emit_rows = self._pending_rows[:self._BATCH_ROWS]
        lookahead_rows = self._pending_rows[self._BATCH_ROWS:]
        n_rows = len(emit_rows)
        batch: List[float] = [s for row in emit_rows for s in row]
        lookahead: List[float] = [s for row in lookahead_rows for s in row]

        overlap_rows = min(self._OVERLAP_ROWS, len(self._history) // data_harmonics)
        overlap_samples = overlap_rows * data_harmonics
        context = self._history[len(self._history) - overlap_samples:] if overlap_samples else []

        native = np.array(context + batch + lookahead, dtype=np.float64)
        resampled = self._upsample(native, chunk_size, data_harmonics)

        overlap_out = overlap_rows * chunk_size
        new_out = n_rows * chunk_size
        result = resampled[overlap_out:overlap_out + new_out]

        self._audio_chunk_output_fifo.push(result.tolist())

        # Carry forward up to _OVERLAP_ROWS rows of context for the next
        # batch. Must be drawn from (context + batch), not just batch, or
        # the history can never grow past _BATCH_ROWS rows regardless of
        # _OVERLAP_ROWS.
        full_history = context + batch
        tail_rows = min(self._OVERLAP_ROWS, overlap_rows + n_rows)
        self._history = full_history[len(full_history) - tail_rows * data_harmonics:]
        self._pending_rows = lookahead_rows

    def _upsample(self, native: np.ndarray, chunk_size: int, data_harmonics: int) -> np.ndarray:
        if self._resample_method == "poly":
            return resample_poly(native, chunk_size, data_harmonics)

        out_len = (len(native) // data_harmonics) * chunk_size

        if self._resample_method == "hold":
            # Zero-order hold: repeat each native sample chunk_size/data_harmonics
            # times. The most basic possible reconstruction — a staircase, not
            # a smooth curve — but with no filter transient to reason about.
            idx = (np.arange(out_len) * len(native)) // out_len
            return native[np.minimum(idx, len(native) - 1)]

        if self._resample_method == "linear":
            # Continuous linear interpolation over the whole batch+context
            # array (not per-row), so there's no seam between symbol rows
            # within a batch. Still a poor reconstruction filter for a ~20x
            # upsample (aliasing/roughness vs "poly"), but simple and has no
            # cross-batch filter-transient behaviour to get wrong.
            native_idx = np.arange(len(native), dtype=np.float64)
            out_idx = np.arange(out_len, dtype=np.float64) * (len(native) - 1) / (out_len - 1)
            return np.interp(out_idx, native_idx, native)

        raise ValueError(f"Unknown resample_method: {self._resample_method!r}")
