from typing import List

import numpy as np

from Settings import Settings
from .AutocorrF0Estimator import AutocorrF0Estimator
from .FFTF0Estimator import FFTF0Estimator
from .PitchQuantizer import quantize_to_chromatic_hz


class F0Tracker:
    """Per-decode-window f0 policy: mode selection (manual/autocorr/fft),
    chromatic quantization, and hold-last-good fallback.

    resolve() is meant to be called once per chunk-aligned decode window with
    that window's pilot segment, so the estimate always describes the exact
    samples it will be used to decode. The estimators run with
    use_pilot_half=False because the caller already isolates the pilot region.
    """

    MODES = ("manual", "autocorr", "fft")

    def __init__(self, settings: Settings):
        self._settings = settings
        self._mode: str = "manual"
        self._manual_f0: float = float(settings.pitch_default_hz)
        self._quantize: bool = False
        self._autocorr = AutocorrF0Estimator(
            f_min_hz=settings.autocorr_f0_min_hz,
            f_max_hz=settings.autocorr_f0_max_hz,
            rms_floor=settings.autocorr_rms_floor,
            corr_threshold=settings.autocorr_corr_threshold,
            use_pilot_half=False,
        )
        self._fft = FFTF0Estimator(
            n_fft=settings.fft_f0_n_fft,
            f_min_hz=settings.fft_f0_min_hz,
            f_max_hz=settings.fft_f0_max_hz,
            rms_floor=settings.fft_rms_floor,
            use_pilot_half=False,
        )
        self._held_f0: float = 0.0
        self._confidence: float = 0.0
        self._pending_manual: float | None = None
        self._pending_skip_windows: int = 0

    # ── policy knobs ─────────────────────────────────────────────────────────
    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown f0 estimator mode: {mode}")
        self._mode = mode
        self._held_f0 = 0.0

    def get_mode(self) -> str:
        return self._mode

    def set_manual_f0(self, f0: float, defer_windows: int = 0) -> None:
        """Sets the manual-mode f0. defer_windows delays adoption by that many
        resolve() calls, so windows already captured at the old pitch are still
        decoded with it (the caller derives the count from its FIFO fill)."""
        f0 = float(f0)
        # Callers (the audio callback) repeat the same value every block; only
        # an actual change may (re-)arm the deferral, otherwise adoption would
        # be pushed back forever.
        target = self._pending_manual if self._pending_manual is not None else self._manual_f0
        if f0 == target:
            return
        if defer_windows <= 0:
            self._manual_f0 = f0
            self._pending_manual = None
        else:
            self._pending_manual = f0
            self._pending_skip_windows = defer_windows

    def set_quantize(self, enabled: bool) -> None:
        self._quantize = bool(enabled)

    # ── state readouts ───────────────────────────────────────────────────────
    @property
    def f0(self) -> float:
        """The f0 last handed out by resolve(); 0.0 before any window resolved."""
        return self._held_f0

    @property
    def confidence(self) -> float:
        return self._confidence

    def has_pitch(self) -> bool:
        """Whether decode currently has a usable pitch. Manual mode always
        does; estimator modes only once at least one window resolved (or the
        held fallback is still alive)."""
        return self._mode == "manual" or self._held_f0 > 0.0

    def reset(self) -> None:
        self._autocorr.reset()
        self._fft.reset()
        self._held_f0 = 0.0
        self._confidence = 0.0

    # ── per-window resolution ────────────────────────────────────────────────
    def resolve(self, pilot_samples: List[float], fs: float, dirty: bool = False) -> float:
        """Resolves the f0 for one decode window from its pilot segment.

        Returns the f0 the window should be decoded with: the fresh estimate
        when one is available, otherwise the last good value (hold), and 0.0
        if no pitch has ever been resolved.

        dirty windows (overlapping gated input) advance the per-window
        bookkeeping but are never estimated — their content is part silence
        and would poison the held value.
        """
        if self._pending_manual is not None:
            if self._pending_skip_windows <= 0:
                self._manual_f0 = self._pending_manual
                self._pending_manual = None
            else:
                self._pending_skip_windows -= 1

        if dirty:
            return self._held_f0

        if self._mode == "manual":
            f_hat = self._manual_f0
            self._confidence = 1.0
        else:
            estimator = self._autocorr if self._mode == "autocorr" else self._fft
            f_hat = estimator.estimate(np.asarray(pilot_samples, dtype=np.float64), fs)
            self._confidence = estimator.confidence

        if self._quantize and f_hat > 0.0:
            f_hat = quantize_to_chromatic_hz(f_hat, self._settings.pitch_quantizer_a4_hz)

        if f_hat > 0.0:
            self._held_f0 = f_hat
        return self._held_f0
