"""Thin matplotlib helpers for thesis figures. matplotlib is imported lazily
so the rest of the package works without it."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ImageFrame = Tuple[bytes, int, int, int]


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_metric_vs_param(rows: List[Dict[str, Any]], x: str, y: str,
                         group_by: Optional[str] = None,
                         logy: bool = False, save: Optional[str] = None):
    """Line plot of a sweep result: metric `y` against parameter `x`, one
    line per distinct value of `group_by` (if given). Repeats are averaged."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = sorted({r.get(group_by) for r in rows}) if group_by else [None]
    for g in groups:
        sub = [r for r in rows if (group_by is None or r.get(group_by) == g)
               and "error" not in r and r.get(y) is not None]
        buckets: Dict[Any, List[float]] = {}
        for r in sub:
            buckets.setdefault(r[x], []).append(float(r[y]))
        xs = sorted(buckets)
        ys = [float(np.mean(buckets[k])) for k in xs]
        ax.plot(xs, ys, marker="o", label=None if g is None else f"{group_by}={g}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if logy:
        ax.set_yscale("log")
    if group_by:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, ax


def plot_spectrogram_pair(carrier: np.ndarray, encoded: np.ndarray, fs: int,
                          n_fft: int = 2048, save: Optional[str] = None):
    """Side-by-side spectrograms of the clean carrier and the encoded signal —
    the classic imperceptibility figure."""
    plt = _plt()
    from scipy.signal import spectrogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sig, title in ((axes[0], carrier, "Carrier (no payload)"),
                           (axes[1], encoded, "Encoded (payload embedded)")):
        f, t, s = spectrogram(sig, fs=fs, nperseg=n_fft, noverlap=n_fft // 2)
        ax.pcolormesh(t, f, 10 * np.log10(s + 1e-12), shading="gouraud", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel("frequency [Hz]")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, axes


def plot_image_pair(reference: ImageFrame, received: Optional[ImageFrame],
                    save: Optional[str] = None):
    """Sent vs decoded image, for qualitative figures."""
    plt = _plt()

    def to_img(frame):
        pixels, w, h, ch = frame
        a = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, ch)
        return a[:, :, 0] if ch == 1 else a

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(to_img(reference), cmap="gray")
    axes[0].set_title("sent")
    if received is not None:
        axes[1].imshow(to_img(received), cmap="gray")
    axes[1].set_title("decoded")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, axes


def plot_f0_track(f0_track: Sequence[float], f0_true: float, fs: int,
                  block_size: int, save: Optional[str] = None):
    """Estimated f0 per block against the true carrier f0."""
    plt = _plt()
    t = np.arange(len(f0_track)) * block_size / fs
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, f0_track, lw=0.8, label="estimated f0")
    ax.axhline(f0_true, color="r", ls="--", label="true f0")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("f0 [Hz]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, ax
