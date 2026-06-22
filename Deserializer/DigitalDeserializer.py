from .Deserializer import Deserializer
from Payload.Payload import Payload
from Payload.SerializedPayload import SerializedPayload


class DigitalDeserializer(Deserializer):
    def deserialize_payload(self, serialized_payload: SerializedPayload) -> Payload:
        pass