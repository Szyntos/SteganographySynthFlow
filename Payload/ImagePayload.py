from Data import Data
from Payload import Payload


class ImagePayload(Payload):
    def __init__(self, size: int):
        super().__init__(size)

    def set_data(self, data: Data):
        pass

    def get_data(self) -> Data:
        pass
