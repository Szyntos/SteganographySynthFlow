from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class DropToleranceConfig:
    tolerance_chunks: int = 3


class DropAction(Enum):
    NORMAL = auto()
    TOLERATE_MOCK = auto()
    RESET_NOW = auto()


class DropTolerance:
    """Tracks consecutive missing/dropped chunks, ported from aoa_cpp_2's
    runtime::DropTolerance.

    Tolerates up to `tolerance_chunks` consecutive missing chunks (caller
    feeds mock/silent data, no reset). The chunk immediately after the
    tolerated run fires RESET_NOW once, so the caller can reset decoder
    state; further consecutive misses go back to TOLERATE_MOCK.
    """

    def __init__(self, cfg: DropToleranceConfig = None):
        self._cfg = cfg if cfg is not None else DropToleranceConfig()
        self._drop_run = 0

    def reset(self) -> None:
        self._drop_run = 0

    @property
    def drop_run(self) -> int:
        return self._drop_run

    @property
    def tolerance(self) -> int:
        return self._cfg.tolerance_chunks

    def push(self, missing_now: bool) -> DropAction:
        if not missing_now:
            self._drop_run = 0
            return DropAction.NORMAL

        self._drop_run += 1
        tol = self._cfg.tolerance_chunks

        if tol > 0 and self._drop_run <= tol:
            return DropAction.TOLERATE_MOCK
        if self._drop_run == tol + 1:
            return DropAction.RESET_NOW
        return DropAction.TOLERATE_MOCK
