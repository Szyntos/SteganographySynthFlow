from Payload.SerializedPayload import SerializedPayload
from Payload.Payload import Payload
from abc import ABC, abstractmethod



class Deserializer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def deserialize_payload(self, serialized_payload: SerializedPayload) -> Payload:
        pass
