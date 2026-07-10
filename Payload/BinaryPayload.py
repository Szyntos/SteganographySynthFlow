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
        self._data = self._encode_with_codec(self._raw_bytes, self._codec)

    def get_data(self) -> List[float]:
        return self._data

    def get_raw_bytes(self) -> bytes:
        return self._raw_bytes
