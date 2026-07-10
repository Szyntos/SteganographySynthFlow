import struct
from typing import List

from Payload import Payload
from Payload.pixel_codec import PixelCodec
from Settings import Settings


class BinaryPayload(Payload):
    def __init__(self, settings: Settings, codec: PixelCodec):
        super().__init__()
        self._settings = settings
        self._codec = codec
        self._raw_bytes: bytes = b""

    def load_from_file(self, file_path: str):
        with open(file_path, "rb") as f:
            self._raw_bytes = f.read()
        self._encode_rows()

    def _encode_rows(self) -> None:
        framed = struct.pack(">I", len(self._raw_bytes)) + self._raw_bytes
        data: List[float] = []
        step = self._codec.chunk_size
        for offset in range(0, len(framed), step):
            data.extend(self._codec.encode_chunk(framed[offset:offset + step]))
        self._data = data

    def get_data(self) -> List[float]:
        return self._data

    def get_raw_bytes(self) -> bytes:
        return self._raw_bytes
