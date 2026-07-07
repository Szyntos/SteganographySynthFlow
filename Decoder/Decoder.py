from typing import List

import numpy as np
from scipy.signal import resample_poly

from AudioChunk import AudioChunk
from Deserializer import Deserializer
from Payload import SymbolRow
from SamplesFifo import SamplesFifo
from Settings import Settings
from .DecodingStrategy import DecodingStrategy


class Decoder:
    # Symbol rows are resampled from their native (data_harmonics-sample)
    # rate up to chunk_size in polyphase batches rather than row-by-row, and
    # a few trailing rows of context are carried across batches. A per-row
    # linear stretch (the old approach) sounds noticeably worse than a WAV
    # dump of the same decoded audio because linear interpolation over a
    # ~10x upsample factor is a poor reconstruction filter; batching lets
    # resample_poly apply a proper band-limited filter instead, at the cost
    # of a little extra latency.
    _BATCH_ROWS: int = 4
    _OVERLAP_ROWS: int = 2

    def __init__(
            self,
            settings: Settings,
            decoding_strategy: DecodingStrategy,
            deserializer: Deserializer,
    ):
        self._settings = settings
        self._decoding_strategy: DecodingStrategy = decoding_strategy
        self._deserializer: Deserializer = deserializer
        self._max_driver_block_size: int = 0
        self._audio_chunk_output_fifo: SamplesFifo = SamplesFifo()
        self._pending_rows: List[List[float]] = []
        self._history: List[float] = []
        self.reconfigure()

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
            if len(self._pending_rows) >= self._BATCH_ROWS:
                self._resample_pending_rows()
        self._deserializer.deserialize_symbols(decoded_symbols)
        return AudioChunk(self._audio_chunk_output_fifo.pop_or_silence(num_samples))

    def _resample_pending_rows(self) -> None:
        data_harmonics = self._settings.data_harmonics
        chunk_size = self._decoding_strategy.get_internal_clock()

        n_rows = len(self._pending_rows)
        batch: List[float] = [s for row in self._pending_rows for s in row]

        overlap_rows = min(self._OVERLAP_ROWS, len(self._history) // data_harmonics)
        overlap_samples = overlap_rows * data_harmonics
        context = self._history[len(self._history) - overlap_samples:] if overlap_samples else []

        resampled = resample_poly(np.array(context + batch, dtype=np.float64), chunk_size, data_harmonics)

        overlap_out = overlap_rows * chunk_size
        new_out = n_rows * chunk_size
        result = resampled[overlap_out:overlap_out + new_out]

        self._audio_chunk_output_fifo.push(result.tolist())

        tail_rows = min(self._OVERLAP_ROWS, n_rows)
        self._history = batch[len(batch) - tail_rows * data_harmonics:]
        self._pending_rows = []
