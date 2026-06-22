from Payload import Payload
from Sink import Sink, SinkBehaviour


class DispatchingSink(Sink[Payload]):
    def __init__(self, sinks: dict[type, Sink], sink_behaviour: SinkBehaviour):
        super().__init__(sink_behaviour)
        self._sinks = sinks

    def push(self, payload: Payload) -> None:
        sink = self._sinks.get(type(payload))
        if sink is None:
            raise ValueError(f"No sink registered for {type(payload)}")
        sink.push(payload)
