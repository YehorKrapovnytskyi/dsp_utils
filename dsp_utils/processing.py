import numpy as np
from scipy.signal import correlate
from typing import Tuple, Union
import types


# Define the Alias
ArrayOrList = Union[np.ndarray, list]


def normalize_rms(x: ArrayOrList, target_rms: float = 0.1) -> np.ndarray:
    """Normalizes signal to a target Root Mean Square amplitude."""
    # Ensure we are working with an array for math
    x_arr = np.asarray(x)
    
    current_rms = np.sqrt(np.mean(np.abs(x_arr) ** 2))
    if current_rms == 0:
        return x_arr
    return x_arr * (target_rms / current_rms)


def pad_signal(signal: np.ndarray, target_length: int, mode: str = "right") -> np.ndarray:
    """Pads or truncates a 1D signal to match a target length.
    
    Parameters:
    -----------
    signal : np.ndarray
        The input 1D signal array.
    target_length : int
        The exact target sample count.
    mode : str, optional
        The padding alignment. Options: "right", "left", "center". Default is "right".
        
    Returns:
    --------
    np.ndarray
        The padded or truncated signal.
    """
    signal_len = len(signal)
    
    # No action needed if lengths match
    if signal_len == target_length:
        return signal
        
    # Truncation Safeguard: Slice the signal if it exceeds the target length
    if signal_len > target_length:
        if mode == "right":
            return signal[:target_length]
        elif mode == "left":
            return signal[-target_length:]
        elif mode == "center":
            # Remove equal amounts from both sides
            start_idx = (signal_len - target_length) // 2
            return signal[start_idx : start_idx + target_length]
        else:
            raise ValueError("Invalid mode. Choose from: 'right', 'left', or 'center'.")

    # Padding Logic
    pad_width = target_length - signal_len
    
    if mode == "right":
        # Add all zeros to the end
        return np.pad(signal, (0, pad_width), mode='constant')
        
    elif mode == "left":
        # Add all zeros to the beginning
        return np.pad(signal, (pad_width, 0), mode='constant')
        
    elif mode == "center":
        # Distribute the padding width across both sides
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left  # Handles odd differences cleanly
        return np.pad(signal, (pad_left, pad_right), mode='constant')
        
    else:
        raise ValueError("Invalid mode. Choose from: 'right', 'left', or 'center'.")


def align_signals(ref: ArrayOrList, target: ArrayOrList) -> Tuple[np.ndarray, int]:
    """
    Cross-correlates target against ref and shifts target to align.
    Returns (aligned_signal, shift_amount).
    """
    # Conversion to array is crucial before correlation
    ref_arr = np.asarray(ref)
    target_arr = np.asarray(target)

    corr = correlate(ref_arr, target_arr, mode='full')
    
    # Zero-lag index for 'full' mode is exactly len(ref)-1
    mid_idx = len(ref_arr) - 1 
    max_idx = np.argmax(np.abs(corr)) 
    shift = max_idx - mid_idx

    aligned = np.roll(target_arr, shift)
    
    # Zero out the 'wrapped' parts from the roll to simulate a shift
    if shift > 0:
        aligned[:shift] = 0
    elif shift < 0:
        aligned[shift:] = 0
    
    return aligned, shift


def add_white_noise(x: ArrayOrList, snr_db: float, signal_rms: float) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal based on a target SNR.
    
    Args:
        x: The input signal (real or complex).
        snr_db: Target Signal-to-Noise Ratio in decibels.
        signal_rms: The Root Mean Square amplitude of the signal. 
                   (Used to calculate the required noise power).
                   
    Returns:
        A NumPy array containing the signal plus noise.
    """
    # Ensure input is an array for mathematical operations
    x_arr = np.asarray(x)
    
    # Calculate noise power based on the SNR formula: SNR_dB = 10 * log10(P_signal / P_noise)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = (signal_rms ** 2) / snr_linear
    
    # Generate noise based on signal type
    if np.iscomplexobj(x_arr):
        # For complex signals, noise power is split between Real and Imaginary parts
        # Total variance = noise_power, so each part gets noise_power / 2
        sigma = np.sqrt(noise_power / 2)
        noise = sigma * (np.random.randn(*x_arr.shape) + 1j * np.random.randn(*x_arr.shape))
    else:
        # For real signals, sigma is just the square root of noise power
        sigma = np.sqrt(noise_power)
        noise = np.random.normal(0, sigma, size=x_arr.shape)
        
    return x_arr + noise


def float_to_int16(x: ArrayOrList) -> np.ndarray:
    """Converts float signal (-1.0 to 1.0) to 16-bit PCM."""
    x_arr = np.asarray(x)
    return (np.clip(x_arr, -1.0, 1.0) * 32767).astype(np.int16)
    

def int16_to_float(x: ArrayOrList) -> np.ndarray:
    """Converts 16-bit PCM to float signal (-1.0 to 1.0)."""
    x_arr = np.asarray(x)
    return (np.clip(x_arr / 32767, -1.0, 1.0)).astype(np.float32)


__all__ = [name for name, obj in globals().items() 
           if isinstance(obj, types.FunctionType) and not name.startswith('_')]