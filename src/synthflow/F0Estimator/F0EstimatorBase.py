from abc import ABC, abstractmethod

import numpy as np


class F0Estimator(ABC):
    """Estimates the fundamental frequency (f0) of an audio chunk, in Hz.

    Implementations return 0.0 when no reliable f0 could be estimated, and
    update `confidence` (roughly in [0, 1]) on every estimate() call so
    callers can judge how trustworthy the returned f0 is, rather than only
    seeing a binary valid/invalid signal.
    """

    def __init__(self):
        self._last_confidence: float = 0.0

    def reset(self) -> None:
        self._last_confidence = 0.0

    @property
    def confidence(self) -> float:
        """Confidence of the most recent estimate() call. 0.0 if no
        estimate has been made yet, or the last one was rejected."""
        return self._last_confidence

    @abstractmethod
    def estimate(self, samples: np.ndarray, fs: float) -> float:
        ...
