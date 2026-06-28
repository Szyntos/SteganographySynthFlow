from typing import List

from Payload import Payload


class TextPayload(Payload):
    def __init__(self):
        super().__init__()

    def load_from_file(self, file_path: str):
        pass

    def get_data(self) -> List[float]:
        pass
