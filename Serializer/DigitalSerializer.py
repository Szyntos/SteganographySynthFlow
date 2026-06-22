from Payload.Payload import Payload
from Payload.SerializedPayload import SerializedPayload
from .Serializer import Serializer


class DigitalSerializer(Serializer):
    def serialize_payload(self, payload: Payload) -> SerializedPayload:
        pass