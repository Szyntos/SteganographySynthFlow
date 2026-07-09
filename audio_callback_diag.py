import os
import time

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "audio_callback.log")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)


def log_callback_event(tag: str, status, duration_s: float, budget_s: float) -> None:
    """Appends a line only when something is worth looking at: a reported
    PortAudio over/underflow, or the callback body running past its budget
    (the audio driver polls at a fixed rate, so overrunning the budget is
    what produces audible clicks)."""
    flags = []
    if getattr(status, "input_underflow", False):
        flags.append("input_underflow")
    if getattr(status, "input_overflow", False):
        flags.append("input_overflow")
    if getattr(status, "output_underflow", False):
        flags.append("output_underflow")
    if getattr(status, "output_overflow", False):
        flags.append("output_overflow")

    over_budget = duration_s > budget_s
    if not flags and not over_budget:
        return

    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{tag}] "
        f"duration_ms={duration_s * 1000:.2f} budget_ms={budget_s * 1000:.2f} "
        f"over_budget={over_budget} flags={','.join(flags) if flags else 'none'}\n"
    )
    with open(_LOG_PATH, "a") as f:
        f.write(line)
