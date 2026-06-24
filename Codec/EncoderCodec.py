from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from Payload.Payload import Payload
from Serializer.Serializer import Serializer

T = TypeVar('T', bound=Payload)


class EncoderCodec(ABC, Generic[T]):
    @abstractmethod
    def serializer(self) -> Serializer[T]:
        pass
