import struct
from abc import ABC, abstractmethod
from typing import List


class Payload(ABC):
    def __init__(self):
        self._data: List[float] = []

    @abstractmethod
    def load_from_file(self, file_path: str):
        pass

    @abstractmethod
    def get_data(self) -> List[float]:
        pass

    @staticmethod
    def _encode_with_codec(raw_bytes: bytes, codec, length_prefixed: bool = True) -> List[float]:
        """Chunk raw bytes through a PixelCodec, one row of floats per chunk.
        When length_prefixed, a 4-byte big-endian length is prepended first,
        so a framed sink can tell real payload from trailing padding — used
        by Binary/Text, which have no fixed size. Image skips it: its fixed
        canvas size makes a length prefix unnecessary."""
        framed = struct.pack(">I", len(raw_bytes)) + raw_bytes if length_prefixed else raw_bytes
        data: List[float] = []
        step = codec.chunk_size
        for offset in range(0, len(framed), step):
            data.extend(codec.encode_chunk(framed[offset:offset + step]))
        return data
