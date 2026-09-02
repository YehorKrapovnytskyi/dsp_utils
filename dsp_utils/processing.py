from typing import Tuple
import numpy as np
from scipy.signal import correlate


def normalize_rms(x: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalizes signal to a target Root Mean Square amplitude."""
    current_rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if current_rms == 0:
        return x
    return x * (target_rms / current_rms)


def pad_signal(
    signal: np.ndarray, target_length: int, mode: str = "right"
) -> np.ndarray:
    """Pads or truncates a 1D signal to match a target length.

    Parameters
    ----------
    signal : np.ndarray
        The input 1D signal array.
    target_length : int
        The exact target sample count.
    mode : str, optional
        The padding alignment. Options: "right", "left", "center". Default is "right".

    Returns
    -------
    np.ndarray
        The padded or truncated signal.
    """
    signal_len = len(signal)

    if signal_len == target_length:
        return signal

    if signal_len > target_length:
        if mode == "right":
            return signal[:target_length]
        elif mode == "left":
            return signal[-target_length:]
        elif mode == "center":
            start_idx = (signal_len - target_length) // 2
            return signal[start_idx : start_idx + target_length]
        else:
            raise ValueError(
                "Invalid mode. Choose from: 'right', 'left', or 'center'."
            )

    pad_width = target_length - signal_len

    if mode == "right":
        return np.pad(signal, (0, pad_width), mode="constant")
    elif mode == "left":
        return np.pad(signal, (pad_width, 0), mode="constant")
    elif mode == "center":
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return np.pad(signal, (pad_left, pad_right), mode="constant")
    else:
        raise ValueError(
            "Invalid mode. Choose from: 'right', 'left', or 'center'."
        )


def align_signals(ref: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, int]:
    """Cross-correlates target against ref and shifts target to align.

    Returns (aligned_signal, shift_amount).
    """
    corr = correlate(ref, target, mode="full")

    mid_idx = len(ref) - 1
    max_idx = np.argmax(np.abs(corr))
    shift = max_idx - mid_idx

    aligned = np.roll(target, shift)

    if shift > 0:
        aligned[:shift] = 0
    elif shift < 0:
        aligned[shift:] = 0

    return aligned, shift


def add_white_noise(
    x: np.ndarray, snr_db: float, signal_rms: float
) -> np.ndarray:
    """Adds Additive White Gaussian Noise (AWGN) to a signal based on a target SNR.

    Parameters
    ----------
    x : np.ndarray
        The input signal (real or complex).
    snr_db : float
        Target Signal-to-Noise Ratio in decibels.
    signal_rms : float
        The Root Mean Square amplitude of the signal.

    Returns
    -------
    np.ndarray
        A NumPy array containing the signal plus noise.
    """
    snr_linear = 10 ** (snr_db / 10)
    noise_power = (signal_rms**2) / snr_linear

    if np.iscomplexobj(x):
        sigma = np.sqrt(noise_power / 2)
        noise = sigma * (
            np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
        )
    else:
        sigma = np.sqrt(noise_power)
        noise = np.random.normal(0, sigma, size=x.shape)

    return x + noise
