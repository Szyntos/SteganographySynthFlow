import cmath
import math
from abc import ABC
from typing import List

import numpy as np

from AudioChunk import AudioChunk
from Framing.SplitLayout import SplitLayout
from Payload import SymbolRow
from SamplesFifo import SamplesFifo
from Settings import Settings


class DecodingStrategy(ABC):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._internal_clock: int = 0
        self._num_rows: int = 0
        self._audio_chunk_input_fifo: SamplesFifo = SamplesFifo()
        self._layout: SplitLayout | None = None
        self._m_state: np.ndarray = np.zeros(0)
        self._alpha: float = settings.decoder_strategy_alpha
        self._mag_threshold: float = 1e-6
        self._analysis_cache_key: tuple | None = None
        self._analysis_cache: tuple | None = None
        self.reconfigure()
        self._f0: float = 440.0

    def reconfigure(self) -> None:
        self._num_rows = self._settings.data_harmonics
        self._internal_clock = self._settings.chunk_size
        self._audio_chunk_input_fifo = SamplesFifo()
        self._m_state = np.zeros(self._num_rows)
        self._analysis_cache_key = None
        self._analysis_cache = None

    def set_f0(self, f0: float):
        self._f0 = f0

    def get_internal_clock(self) -> int:
        return self._internal_clock

    def get_input_fifo_size(self) -> int:
        return self._audio_chunk_input_fifo.get_size()

    def decode_samples(self, input_samples: AudioChunk, num_samples: int) -> List[SymbolRow]:
        self._audio_chunk_input_fifo.push(input_samples.get_samples())

        decoded_symbols: List[SymbolRow] = []

        while self._audio_chunk_input_fifo.can_read(self._internal_clock):
            to_decode: List[float] = self._audio_chunk_input_fifo.pop_or_empty(self._internal_clock)
            decoded_symbols.append(self._decode(to_decode))
        return decoded_symbols

    def _get_analysis_arrays(self, fs: float, window_size: int):
        # The projection matrices depend on every term below, not just f0: a
        # key that omits the window geometry silently returns matrices built
        # for the previous chunk_size/layout.
        total_harmonics = self._settings.total_harmonics
        key = (
            self._f0, fs, window_size, total_harmonics,
            self._layout.pilot_start, self._layout.data_start,
        )
        if self._analysis_cache_key == key and self._analysis_cache is not None:
            return self._analysis_cache

        nyquist = 0.5 * fs

        # Use the largest integer-cycle window length that fits in window_size.
        # This ensures harmonics are mutually orthogonal, eliminating cross-harmonic
        # leakage that otherwise biases the phase measurement when f0*window_size/fs is
        # non-integer (e.g. f0=500, window_size=240, fs=48000 -> 2.5 cycles, not integer).
        cycles_per_period = max(1, int(round(fs / self._f0)))
        n_cycles = window_size // cycles_per_period
        if n_cycles < 1:
            # f0 too low to fit even one cycle in window_size; use all available samples
            analysis_len = window_size
        else:
            analysis_len = n_cycles * cycles_per_period
        win_start = (window_size - analysis_len) // 2  # centre window in each segment

        n = np.arange(analysis_len, dtype=np.float64)
        hann_win = 0.5 - 0.5 * np.cos(2.0 * math.pi * n / (analysis_len - 1)) if analysis_len > 1 else np.ones(1)

        t_p = n + self._layout.pilot_start + win_start
        t_d = n + self._layout.data_start + win_start

        # omega_k = 2*pi*(k+1)*f0/fs for k in [0..total_harmonics-1]
        harmonic_nums = np.arange(1, total_harmonics + 1, dtype=np.float64)
        freqs = harmonic_nums * self._f0
        valid_mask = (freqs > 0.0) & (freqs <= nyquist)
        omegas = np.where(valid_mask, 2.0 * math.pi * freqs / fs, 0.0)

        # Precompute the DFT projection matrices; these only depend on f0/fs/window_size.
        phase_p = np.outer(omegas, t_p)
        phase_d = np.outer(omegas, t_d)
        proj_p = np.exp(-1j * phase_p)
        proj_d = np.exp(-1j * phase_d)

        self._analysis_cache_key = key
        self._analysis_cache = (analysis_len, win_start, hann_win, valid_mask, proj_p, proj_d)
        return self._analysis_cache

    def _decode(self, samples: List[float]) -> SymbolRow:
        if self._f0 <= 0.0:
            return SymbolRow(self._m_state.tolist())

        fs = float(self._settings.fs_out)
        data_offset = self._settings.data_offset
        data_harmonics = self._settings.data_harmonics
        phase_range = float(self._settings.phase_range)
        total_harmonics = self._settings.total_harmonics
        window_size = self._internal_clock // self._layout.phases

        analysis_len, win_start, hann_win, valid_mask, proj_p, proj_d = self._get_analysis_arrays(fs, window_size)

        samples_arr = np.asarray(samples, dtype=np.float64)
        pilot_start = self._layout.pilot_start + win_start
        data_start = self._layout.data_start + win_start
        xp = samples_arr[pilot_start:pilot_start + analysis_len] * hann_win
        xd = samples_arr[data_start:data_start + analysis_len] * hann_win

        # Vectorised DFT projections: (H, analysis_len) outer product, precomputed above
        Z_p = (xp * proj_p).sum(axis=1)
        Z_d = (xd * proj_d).sum(axis=1)
        Z_p = np.where(valid_mask, Z_p, 0j)
        Z_d = np.where(valid_mask, Z_d, 0j)

        ap = np.abs(Z_p)
        ad = np.abs(Z_d)

        # Bias phase from non-data-band harmonics
        non_data = np.ones(total_harmonics, dtype=bool)
        non_data[data_offset:data_offset + data_harmonics] = False
        bias_mask = valid_mask & non_data & (ap >= self._mag_threshold) & (ad >= self._mag_threshold)

        sum_unit = complex(0.0, 0.0)
        if bias_mask.any():
            prod_bias = Z_d[bias_mask] * np.conj(Z_p[bias_mask])
            w = ap[bias_mask] * ad[bias_mask]
            sum_unit = np.sum(w * np.exp(1j * np.angle(prod_bias)))

        phi_bias = cmath.phase(sum_unit) if abs(sum_unit) > 0.0 else 0.0

        # Decode each data harmonic
        n_data = min(data_harmonics, total_harmonics - data_offset)
        h_idx = np.arange(data_offset, data_offset + n_data)

        data_valid = valid_mask[h_idx] & (ap[h_idx] >= self._mag_threshold) & (ad[h_idx] >= self._mag_threshold)

        prod_data = Z_d[h_idx] * np.conj(Z_p[h_idx])
        delta = np.angle(prod_data)
        delta_centered = (delta - phi_bias + math.pi) % (2.0 * math.pi) - math.pi
        m_hat = delta_centered / phase_range

        out = np.where(
            data_valid,
            (1.0 - self._alpha) * self._m_state[:n_data] + self._alpha * m_hat,
            self._m_state[:n_data],
        )

        self._m_state[:n_data] = out

        # Pad to data_harmonics if total_harmonics - data_offset < data_harmonics
        if n_data < data_harmonics:
            out = np.concatenate([out, self._m_state[n_data:data_harmonics]])

        return SymbolRow(out.tolist())
