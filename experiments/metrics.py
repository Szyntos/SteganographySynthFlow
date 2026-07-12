"""Metrics for thesis experiments.

Three families:
  1. Payload fidelity — did the hidden data survive? (BER, PSNR/SSIM,
     character error rate, audio SNR/RMSE)
  2. Imperceptibility — how much does embedding disturb the carrier?
     (embedding SNR, log-spectral distance)
  3. System behaviour — throughput, latency, real-time headroom, f0-tracking
     accuracy.

All functions are numpy-only (plus scipy) and take plain arrays/bytes so they
work on RunResult fields or on anything else.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

ImageFrame = Tuple[bytes, int, int, int]


# ── 1. payload fidelity: bits & bytes ───────────────────────────────────────

def bit_error_rate(reference: bytes, received: Optional[bytes]) -> float:
    """Fraction of payload bits wrong. A missing/short received payload counts
    its absent bits as errors, so 'never synced' scores 1.0, not NaN."""
    if not reference:
        return 0.0
    ref = np.frombuffer(reference, dtype=np.uint8)
    if received is None:
        return 1.0
    rec = np.frombuffer(received[: len(reference)], dtype=np.uint8)
    errors = int(np.unpackbits(ref[: len(rec)] ^ rec).sum())
    errors += 8 * (len(ref) - len(rec))  # truncated tail: all bits lost
    return errors / (8 * len(ref))


def byte_error_rate(reference: bytes, received: Optional[bytes]) -> float:
    if not reference:
        return 0.0
    if received is None:
        return 1.0
    rec = received[: len(reference)]
    errors = sum(a != b for a, b in zip(reference, rec)) + (len(reference) - len(rec))
    return errors / len(reference)


# ── 1. payload fidelity: images ─────────────────────────────────────────────

def _to_array(frame: ImageFrame) -> np.ndarray:
    pixels, w, h, ch = frame
    return np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, ch).astype(np.float64)


def image_mse(reference: ImageFrame, received: Optional[ImageFrame]) -> float:
    if received is None:
        return float("inf")
    return float(np.mean((_to_array(reference) - _to_array(received)) ** 2))


def image_psnr_db(reference: ImageFrame, received: Optional[ImageFrame]) -> float:
    mse = image_mse(reference, received)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse)) if np.isfinite(mse) else float("-inf")


def image_ssim(reference: ImageFrame, received: Optional[ImageFrame], window: int = 7) -> float:
    """Mean SSIM with a uniform window, averaged over channels."""
    if received is None:
        return 0.0
    a, b = _to_array(reference), _to_array(received)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_channels = []
    for ch in range(a.shape[2]):
        x, y = a[:, :, ch], b[:, :, ch]
        mx, my = uniform_filter(x, window), uniform_filter(y, window)
        vx = uniform_filter(x * x, window) - mx * mx
        vy = uniform_filter(y * y, window) - my * my
        cxy = uniform_filter(x * y, window) - mx * my
        s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
        ssim_channels.append(float(s.mean()))
    return float(np.mean(ssim_channels))


# ── 1. payload fidelity: text ───────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_error_rate(reference: str, received: Optional[str]) -> float:
    if not reference:
        return 0.0
    if received is None:
        return 1.0
    return levenshtein(reference, received) / len(reference)


# ── 1. payload fidelity: audio ──────────────────────────────────────────────

def audio_rmse(reference: Sequence[float], received: Sequence[float]) -> float:
    n = min(len(reference), len(received))
    d = np.asarray(reference[:n]) - np.asarray(received[:n])
    return float(np.sqrt(np.mean(d ** 2))) if n else float("inf")


def audio_snr_db(reference: Sequence[float], received: Sequence[float]) -> float:
    """SNR of the recovered audio payload vs the ideal decoder output."""
    n = min(len(reference), len(received))
    ref = np.asarray(reference[:n])
    err = ref - np.asarray(received[:n])
    p_err = float(np.mean(err ** 2))
    if p_err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(np.mean(ref ** 2) / p_err))


# ── 2. imperceptibility (encoded signal vs clean carrier) ───────────────────

def embedding_snr_db(carrier: np.ndarray, encoded: np.ndarray) -> float:
    """Carrier-to-embedding-distortion ratio: higher = harder to hear the
    hidden data. Compare RunResult.encoded against render_carrier(config)."""
    n = min(len(carrier), len(encoded))
    return audio_snr_db(carrier[:n], encoded[:n])


def log_spectral_distance_db(carrier: np.ndarray, encoded: np.ndarray,
                             n_fft: int = 4096) -> float:
    """Mean log-spectral distance between carrier and encoded signal —
    a crude spectral audibility proxy complementing embedding SNR."""
    n = (min(len(carrier), len(encoded)) // n_fft) * n_fft
    if n == 0:
        return float("nan")
    a = np.abs(np.fft.rfft(carrier[:n].reshape(-1, n_fft), axis=1)) + 1e-12
    b = np.abs(np.fft.rfft(encoded[:n].reshape(-1, n_fft), axis=1)) + 1e-12
    return float(np.mean(np.sqrt(np.mean((20 * np.log10(a / b)) ** 2, axis=1))))


# ── 3. system behaviour ─────────────────────────────────────────────────────

def raw_bitrate_bps(settings) -> float:
    """Physical-layer bitrate: data bits carried per second of audio,
    before framing/sync overhead."""
    return settings.bits_per_chunk * settings.fs_out / settings.chunk_size


def goodput_bps(payload_bytes: int, first_frame_latency_s: Optional[float]) -> float:
    """Effective delivered bitrate: payload size over time to first complete
    frame (includes sync overhead and startup)."""
    if first_frame_latency_s is None or first_frame_latency_s <= 0:
        return 0.0
    return 8.0 * payload_bytes / first_frame_latency_s


def f0_tracking_error_hz(f0_true: float, f0_track: Sequence[float]) -> float:
    """Mean absolute f0 estimation error over blocks where a pitch was held."""
    track = np.asarray([f for f in f0_track if f > 0.0])
    return float(np.mean(np.abs(track - f0_true))) if track.size else float("nan")
