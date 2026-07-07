from typing import List

from PIL import Image

from Payload import Payload
from Payload.pixel_codec import PixelCodec
from Settings import Settings


class ImagePayload(Payload):
    def __init__(self, settings: Settings, codec: PixelCodec):
        super().__init__()
        self._settings = settings
        self._codec = codec
        self._pixel_bytes: bytes = b""

    def load_from_file(self, file_path: str):
        width = self._settings.image_target_w
        height = self._settings.image_target_h
        channels = self._settings.image_channels
        if width <= 0 or height <= 0:
            raise ValueError("ImagePayload: bad target size")
        if channels not in (1, 3):
            raise ValueError("ImagePayload: channels must be 1 or 3")

        with Image.open(file_path) as image:
            image = image.convert("L" if channels == 1 else "RGB")
            image = image.resize((width, height), Image.BILINEAR)
            self._pixel_bytes = image.tobytes()

        self._encode_rows()

    def _encode_rows(self) -> None:
        data: List[float] = []
        step = self._codec.chunk_size
        for offset in range(0, len(self._pixel_bytes), step):
            data.extend(self._codec.encode_chunk(self._pixel_bytes[offset:offset + step]))
        self._data = data

    def get_data(self) -> List[float]:
        return self._data

    def get_pixel_bytes(self) -> bytes:
        return self._pixel_bytes
