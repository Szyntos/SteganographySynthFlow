from typing import List

from AudioChunk import AudioChunk


class AudioChunkSink:
    def __init__(self, startup_threshold: int = 0):
        self._samples: List[float] = []
        self._startup_threshold: int = startup_threshold
        self._started: bool = startup_threshold == 0

    def get_size(self) -> int:
        return len(self._samples)

    def has_started(self) -> bool:
        return self._started

    def push(self, audio_chunk: AudioChunk) -> None:
        self._samples += audio_chunk.get_samples()

        if not self._started and len(self._samples) >= self._startup_threshold:
            self._started = True

    def get_n_samples_or_0(self, n: int) -> List[float]:
        if not self._started:
            return []

        if len(self._samples) >= n:
            result = self._samples[:n]
            self._samples = self._samples[n:]
            return result

        return []
