from Payload import Payload
from SerializedPayload import SerializedPayload
from Serializer.Serializer import Serializer


class DigitalSerializer(Serializer):
    def serialize_payload(self, payload: Payload) -> SerializedPayload:
        pass