from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from Payload.SerializedPayload import SerializedPayload
from Payload.Payload import Payload
from SerializerMode import SerializerMode

T = TypeVar('T', bound=Payload)


class Deserializer(ABC, Generic[T]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_float: int = 1):
        self._serializer_mode: SerializerMode = serializer_mode
        self._bits_per_float: int = bits_per_float

    def set_bits_per_float(self, bits_per_float: int):
        self._bits_per_float = bits_per_float

    def set_serializer_mode(self, serializer_mode: SerializerMode):
        self._serializer_mode = serializer_mode

    @abstractmethod
    def deserialize_payload(self, serialized_payload: SerializedPayload) -> T:
        pass
