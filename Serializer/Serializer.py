from Payload.SerializedPayload import SerializedPayload
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from Payload.Payload import Payload
from SerializerMode import SerializerMode

T = TypeVar('T', bound=Payload)

class Serializer(ABC, Generic[T]):
    def __init__(self, serializer_mode: SerializerMode, bits_per_symbol: int = 1):
        self._serializer_mode: SerializerMode = serializer_mode
        self._bits_per_symbol: int = bits_per_symbol

    @abstractmethod
    def serialize_payload(self, payload: T) -> SerializedPayload:
        pass
