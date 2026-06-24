from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from Deserializer.Deserializer import Deserializer
from Payload.Payload import Payload
from Sink.Sink import Sink

T = TypeVar('T', bound=Payload)


class DecoderCodec(ABC, Generic[T]):
    @abstractmethod
    def deserializer(self) -> Deserializer[T]:
        pass

    @abstractmethod
    def sink(self) -> Sink[T]:
        pass
