from SerializedPayload import SerializedPayload
from Payload import Payload


class Serializer:
    def __init__(self, payload: Payload):
        self._payload: Payload = payload

    def set_payload(self, payload: Payload) -> None:
        self._payload = payload

    def serialize_payload(self) -> SerializedPayload:
        pass
