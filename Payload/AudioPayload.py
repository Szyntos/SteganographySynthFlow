from typing import List
import math
from Payload import Payload


class AudioPayload(Payload):
    def __init__(self):
        super().__init__()

    def load_from_file(self, file_path: str):
        pass

    def get_data(self) -> List[float]:
        return [math.sin(i/200) + math.sin(i/400)  for i in range(1000)]