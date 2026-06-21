class MidiInput:
    def __init__(self):
        self.note: float = 0.0
        pass
    def on_play(self, function) -> None:
        function(self.note)
