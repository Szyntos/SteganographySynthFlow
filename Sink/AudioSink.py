from Framing import FramingSyncController
from Payload import AudioPayload
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class AudioSink(Sink[AudioPayload]):
    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        super().__init__(framing_sync_controller, sink_behaviour)

    def push(self, payload: AudioPayload) -> None:
        pass
