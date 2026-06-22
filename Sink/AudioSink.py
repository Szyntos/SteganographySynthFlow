from Payload import AudioPayload
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class AudioSink(Sink[AudioPayload]):
    def __init__(self):
        super().__init__(SinkBehaviour.LIVE)

    def push(self, payload: AudioPayload) -> None:
        pass
