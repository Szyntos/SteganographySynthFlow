from Payload.SerializedPayload import SerializedPayload
from Payload.Payload import Payload
from abc import ABC, abstractmethod

class Serializer(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def serialize_payload(self, payload: Payload) -> SerializedPayload:
        pass
