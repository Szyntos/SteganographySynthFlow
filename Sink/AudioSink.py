from Payload import AudioPayload
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class AudioSink(Sink[AudioPayload]):
    def __init__(self, sink_behaviour: SinkBehaviour):
        super().__init__(sink_behaviour)

    def push(self, payload: AudioPayload) -> None:
        pass
