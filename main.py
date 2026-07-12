"""Headless encode→decode round trip, plotted.

The pipeline itself lives in exp/harness.py; this is just the plot.
"""

import matplotlib.pyplot as plt

from exp.harness import RoundTrip, run_round_trip


def graph_round_trip(rt: RoundTrip) -> None:
    startup_lag = rt.startup_lag
    expected = rt.expected
    diff_x = range(startup_lag, startup_lag + len(expected))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax1.plot(rt.encoded)
    ax1.set_title("Encoded")
    ax1.set_ylabel("Amplitude")

    ax2.plot(rt.decoded, label="Decoded")
    ax2.plot(diff_x, expected, label="Expected", alpha=0.7)
    ax2.axvline(
        x=startup_lag, color="r", linestyle="--", alpha=0.5,
        label=f"Startup lag ({startup_lag} samples)",
    )
    ax2.set_title("Decoded")
    ax2.set_ylabel("Amplitude")
    ax2.legend()

    ax3.plot(diff_x, rt.diff, color="purple")
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_title(f"Diff (Decoded − Expected)   RMSE = {rt.rmse():.4f}")
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Sample index")

    plt.tight_layout()
    plt.show()


def main() -> None:
    rt = run_round_trip(f0=500.0, num_chunks=160)
    print(f"RMSE (decoded − expected): {rt.rmse():.6f}")
    graph_round_trip(rt)


if __name__ == "__main__":
    main()
