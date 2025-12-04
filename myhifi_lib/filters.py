"""
Audio filters and corrections.

General-purpose functions:
- apply_frequency_slope() - Apply dB/octave slope to frequency range (general DSP)

Time-domain filters (applied to audio signals):
- RIAA equalization (phono-specific)
- XG7001 custom bass filter (phono test record specific)

Frequency-domain corrections (applied to freq/amp arrays):
- STR100 bass correction (phono test record specific, uses apply_frequency_slope)
"""

import numpy as np
from scipy.signal import lfilter


def riaaiir(signal, Fs, mode, inverse=False):
    """
    Apply RIAA equalization using IIR filters.
    
    This is a time-domain filter applied to the raw audio signal before FFT analysis.
    
    Parameters:
    - signal: audio signal (numpy array, can be mono or stereo)
    - Fs: sample rate (currently only 96000 Hz supported)
    - mode: 1=bass only, 2=treble only, 3=both
    - inverse: if True, inverts the filter (de-emphasis becomes emphasis, etc.)
    
    Returns:
    - filtered signal (same shape as input)
    
    Notes:
    - Mode 1 (bass): Applies bass rolloff/shelving filter
    - Mode 2 (treble): Applies treble de-emphasis/emphasis
    - Mode 3 (both): Applies full RIAA equalization curve
    """
    if Fs == 96000:
        # Treble filter coefficients
        at = [1, -0.66168391, -0.18158841]
        bt = [0.1254979638905360, 0.0458786797031512, 0.0018820452752401]
        
        # Bass filter coefficients
        ars = [1, -0.60450091, -0.39094593]
        brs = [0.90861261463964900, -0.52293147388301200, -0.34491369168550900]
    else:
        raise ValueError(f"RIAA filter only implemented for 96kHz sample rate, got {Fs}Hz")
    
    # Invert if requested (swap numerator and denominator)
    if inverse:
        at, bt = bt, at
        ars, brs = brs, ars
    
    # Apply filters based on mode
    if mode == 1:
        signal = lfilter(brs, ars, signal)
    elif mode == 2:
        signal = lfilter(bt, at, signal)
    elif mode == 3:
        signal = lfilter(bt, at, signal)
        signal = lfilter(brs, ars, signal)
    else:
        raise ValueError(f"Invalid RIAA mode {mode}, must be 1, 2, or 3")
    
    return signal


def normxg7001(signal, Fs):
    """
    Apply XG7001 custom bass filter.
    
    This is a time-domain filter for the Denon XG-7001 test record.
    Based on @stereoplay filter specification.
    
    Parameters:
    - signal: audio signal (numpy array)
    - Fs: sample rate (currently only 96000 Hz supported)
    
    Returns:
    - filtered signal (same shape as input)
    """
    if Fs == 96000:
        b = [1.0080900, -0.9917285, 0]
        a = [1, -0.9998364, 0]
        signal = lfilter(b, a, signal)
    else:
        raise ValueError(f"XG7001 filter only implemented for 96kHz sample rate, got {Fs}Hz")
    
    return signal


def apply_frequency_slope(freq_array, amp_array, fmin, fmax, slope_db_per_octave):
    """
    Apply a logarithmic slope correction to a frequency range.
    
    General-purpose function for applying dB/octave corrections to frequency response data.
    Useful for bass/treble tilt corrections, test record compensations, etc.
    
    Parameters:
    - freq_array: frequency values (Hz) - list or numpy array
    - amp_array: amplitude values (dB) - list or numpy array
    - fmin: start frequency for correction (Hz)
    - fmax: reference frequency for correction (Hz)
    - slope_db_per_octave: slope in dB/octave (negative = rolloff, positive = boost)
    
    Returns:
    - corrected amplitude array (same type as input)
    
    Example:
    >>> # Apply -6 dB/octave rolloff from 40Hz to 500Hz
    >>> amp_corrected = apply_frequency_slope(freq, amp, 40, 500, -6.02)
    """
    from .analysis import find_nearest
    
    # Convert to numpy for processing if needed
    was_list = isinstance(amp_array, list)
    freq = np.asarray(freq_array)
    amp = np.asarray(amp_array)
    
    # Find indices for the correction range
    idx_min = find_nearest(freq, fmin)
    idx_max = find_nearest(freq, fmax)
    
    # Apply logarithmic slope correction to each frequency in range
    for x in range(idx_min, idx_max):
        correction = 20 * np.log10(((freq[x]) / fmax) ** ((slope_db_per_octave / 20) / np.log10(2)))
        amp[x] = amp[x] + correction
    
    # Return same type as input
    return amp.tolist() if was_list else amp


def normstr100(freq_array, amp_array):
    """
    Apply STR100 bass correction to frequency response data.
    
    This applies a -6.02 dB/octave rolloff correction from 40Hz to 500Hz,
    compensating for the STR100 phono test record's bass characteristics.
    
    This is a convenience wrapper around apply_frequency_slope() with
    STR100-specific parameters.
    
    Parameters:
    - freq_array: frequency values (Hz) - list or numpy array
    - amp_array: amplitude values (dB) - list or numpy array
    
    Returns:
    - corrected amplitude array (same type as input)
    """
    return apply_frequency_slope(freq_array, amp_array, 40, 500, -6.02)
