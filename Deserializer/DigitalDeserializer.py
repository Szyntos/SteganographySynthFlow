from .Deserializer import Deserializer
from Payload import Payload
from SerializedPayload import SerializedPayload


class DigitalDeserializer(Deserializer):
    def deserialize_payload(self, serialized_payload: SerializedPayload) -> Payload:
        pass