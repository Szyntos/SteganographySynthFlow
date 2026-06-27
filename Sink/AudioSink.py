import math

from AudioChunk import AudioChunk
from Framing import FramingSyncController
from Payload import AudioPayload
from .Sink import Sink
from .SinkBehaviour import SinkBehaviour


class AudioSink(Sink):
    def __init__(self, framing_sync_controller: FramingSyncController, sink_behaviour: SinkBehaviour):
        super().__init__(framing_sync_controller, sink_behaviour)

    def push(self, payload: AudioPayload) -> None:
        pass

    def get_audio_chunk(self, num_samples: int) -> AudioChunk:
        return AudioChunk([math.sin(i) for i in range(num_samples)]) # will return chunk of audio from the saved payloads