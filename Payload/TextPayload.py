import struct
from typing import List

from Payload import Payload
from Payload.pixel_codec import PixelCodec
from Settings import Settings


class TextPayload(Payload):
    def __init__(self, settings: Settings, codec: PixelCodec):
        super().__init__()
        self._settings = settings
        self._codec = codec
        self._text: str = ""

    def load_from_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            self._text = f.read()
        self._encode(self._text.encode("utf-8"))

    def load_from_string(self, text: str) -> None:
        self._text = text
        self._encode(text.encode("utf-8"))

    def _encode(self, raw_bytes: bytes) -> None:
        framed = struct.pack(">I", len(raw_bytes)) + raw_bytes
        data: List[float] = []
        step = self._codec.chunk_size
        for offset in range(0, len(framed), step):
            data.extend(self._codec.encode_chunk(framed[offset:offset + step]))
        self._data = data

    def get_data(self) -> List[float]:
        return self._data

    def get_text(self) -> str:
        return self._text
