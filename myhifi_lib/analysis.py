"""
Audio frequency response analysis using swept sine signals.

Core functions:
- FFT-based frequency response extraction from swept sine signals
- Harmonic distortion detection (2nd and 3rd harmonics)
- Crosstalk measurement for stereo signals
- Utility functions for data processing

Works with any swept sine source: test records, signal generators, software sweeps, etc.
"""

import numpy as np
from scipy.signal import sosfiltfilt, iirfilter


def ft_window(n):
    """
    Generate a flat-top window (Matlab compatible).
    
    Flat-top windows provide excellent amplitude accuracy for FFT analysis,
    which is critical for precise frequency response measurement.
    
    Parameters:
    - n: window length (samples)
    
    Returns:
    - window coefficients (numpy array)
    """
    a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
    x = np.arange(n)
    w = (a0 - a1*np.cos(2*np.pi*x/(n-1)) + a2*np.cos(4*np.pi*x/(n-1)) - 
         a3*np.cos(6*np.pi*x/(n-1)) + a4*np.cos(8*np.pi*x/(n-1)))
    return w


def find_nearest(array, value):
    """
    Find index of nearest value in a sorted array using binary search.
    
    Parameters:
    - array: sorted array (list or numpy array)
    - value: target value
    
    Returns:
    - index of nearest value
    """
    array = np.asarray(array)
    idx = np.searchsorted(array, value)
    
    # Handle edge cases
    if idx == 0:
        return 0
    if idx == len(array):
        return len(array) - 1
    
    # Check which is actually closer: idx or idx-1
    if abs(array[idx] - value) < abs(array[idx-1] - value):
        return idx
    else:
        return idx - 1


def ordersignal(signal, Fs):
    """
    Determine sweep direction (up or down) and flip if needed.
    
    Analyzes the start and end of the signal to determine if it's a
    rising or falling frequency sweep, and flips the signal if falling.
    
    Parameters:
    - signal: audio signal (can be mono or stereo)
    - Fs: sample rate
    
    Returns:
    - signal: possibly flipped signal (rising sweep)
    - minf: starting frequency bin
    - maxf: ending frequency bin
    """
    # Ensure signal is 2D
    if len(signal.shape) == 1:
        signal = np.expand_dims(signal, axis=0)
    
    # Analyze start and end of signal
    F = int(Fs / 100)
    win = ft_window(F)
    
    # Find dominant frequency at start
    y = abs(np.fft.rfft(signal[0, 0:F] * win))
    minf = np.argmax(y)
    
    # Find dominant frequency at end
    y = abs(np.fft.rfft(signal[0, len(signal[0]) - F:len(signal[0])] * win))
    maxf = np.argmax(y)
    
    # Flip if sweep is descending
    if maxf < minf:
        maxf, minf = minf, maxf
        signal = np.flipud(signal)
    
    return signal, minf, maxf


def normalize_to_frequency(freq_array, amp_array, normalize_freq):
    """
    Normalize amplitude data to 0dB at a specific frequency.
    
    Parameters:
    - freq_array: frequency values (Hz) - list or numpy array
    - amp_array: amplitude values (dB) - list or numpy array
    - normalize_freq: frequency to set as 0dB reference
    
    Returns:
    - normalized amplitude array (same type as input)
    """
    was_list = isinstance(amp_array, list)
    freq = np.asarray(freq_array)
    amp = np.asarray(amp_array)
    
    idx = find_nearest(freq, normalize_freq)
    offset = amp[idx]
    normalized = amp - offset
    
    return normalized.tolist() if was_list else normalized


def analyze_sweep(signal, Fs, start_f=None, end_f=20000, smoothing=0.5):
    """
    Extract frequency response, harmonics, and crosstalk from swept sine measurement.
    
    This function processes a monotonic frequency sweep (rising or falling) and extracts:
    - Primary frequency response
    - Crosstalk (for stereo signals)
    - 2nd harmonic distortion
    - 3rd harmonic distortion
    
    The sweep can be from any source - test records, signal generators, software sweeps, etc.
    
    Parameters:
    - signal: audio signal (mono or stereo, numpy array)
    - Fs: sample rate (Hz)
    - start_f: start frequency for output (Hz), None = no limit
    - end_f: end frequency for output (Hz), None = no limit
    - smoothing: lowpass filter cutoff for noise reduction (default 0.5, use lower 
                 values like 0.02 for busy comparison plots)
    
    Returns dict with keys:
    - freq: frequency values for primary response
    - amp: amplitude values for primary response (dB)
    - crosstalk_freq: frequency values for crosstalk
    - crosstalk_amp: amplitude values for crosstalk (dB)
    - harmonic2_freq: frequency values for 2nd harmonic
    - harmonic2_amp: amplitude values for 2nd harmonic (dB)
    - harmonic3_freq: frequency values for 3rd harmonic
    - harmonic3_amp: amplitude values for 3rd harmonic (dB)
    
    Empty lists are returned for missing data (e.g., crosstalk for mono signals).
    
    Notes:
    - All amplitude values are in dB (20*log10)
    - Signal can be mono (1D) or stereo (2D)
    - For stereo: channel 0 is primary, channel 1 is crosstalk
    - Processing is done in frequency chunks with different bin sizes
    - A smoothing lowpass filter is applied to reduce noise
    """
    
    def bin_and_average(f, a, minf, maxf, fstep):
        """
        Bin frequency/amplitude data and average within bins.
        
        Takes scattered frequency measurements and bins them into regular
        frequency intervals, averaging all measurements within each bin.
        """
        f, a = np.array(f), np.array(a)
        
        bins = np.arange(minf, maxf + fstep, fstep)
        indices = np.digitize(f, bins) - 1
        f_out, a_out = [], []
        
        for i, bin_center in enumerate(bins):
            mask = (indices == i)
            if np.any(mask):
                f_out.append(bin_center)
                a_out.append(20 * np.log10(np.mean(a[mask])))
        
        return f_out, a_out
    
    
    def rfft(signal, Fs, minf, maxf, fstep):
        """
        Perform FFT analysis on swept sine to extract frequency response and harmonics.
        
        Uses overlapping windowed FFTs to track the fundamental frequency and its harmonics
        as the sweep progresses through the frequency range.
        """
        freq, amp, freqx, ampx, freq2h, amp2h, freq3h, amp3h = [], [], [], [], [], [], [], []
        
        F = int(Fs / fstep)
        win = ft_window(F)
        
        # Ensure signal is 2D
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)
        
        # Process signal in non-overlapping windows
        for x in range(0, signal.shape[1] - F, F):
            # FFT of primary channel
            y0 = abs(np.fft.rfft(signal[0, x:x + F] * win))
            f0 = np.argmax(y0)  # Find fundamental frequency bin
            
            # Store fundamental if in range
            if f0 >= minf/fstep and f0 <= maxf/fstep:
                freq.append(f0 * fstep)
                amp.append(y0[f0])
            
            # Store 2nd harmonic if it fits in spectrum
            if 2*f0 < F/2 - 2 and f0 > minf/fstep and f0 < maxf/fstep:
                f2 = np.argmax(y0[(2*f0) - 2:(2*f0) + 2])
                freq2h.append(f0 * fstep)
                amp2h.append(y0[2*f0 - 2 + f2])
            
            # Store 3rd harmonic if it fits in spectrum
            if 3*f0 < F/2 - 2 and f0 > minf/fstep and f0 < maxf/fstep:
                f3 = np.argmax(y0[(3*f0) - 2:(3*f0) + 2])
                freq3h.append(f0 * fstep)
                amp3h.append(y0[3*f0 - 2 + f3])
            
            # Process second channel for crosstalk (if stereo)
            if signal.shape[0] > 1:
                y1 = abs(np.fft.rfft(signal[1, x:x + F] * win))
                f1 = np.argmax(y1)
                if f0 >= minf/fstep and f0 <= maxf/fstep:  # Use primary sweep f range
                    freqx.append(f1 * fstep)
                    ampx.append(y1[f1])
            else:
                ampx = []
                freqx = []
        
        return freq, amp, freqx, ampx, freq2h, amp2h, freq3h, amp3h
    
    
    def process_chunk(signal, Fs, fmin, fmax, step, offset):
        """
        Process one frequency chunk with specific bin size and offset.
        
        Different frequency ranges use different FFT bin sizes for optimal
        frequency resolution vs. time resolution tradeoff.
        """
        f, a, fx, ax, f2, a2, f3, a3 = rfft(signal, Fs, fmin, fmax, step)
        
        f, a = bin_and_average(f, a, fmin, fmax, step)
        fx, ax = bin_and_average(fx, ax, fmin, fmax, step)
        f2, a2 = bin_and_average(f2, a2, fmin, fmax, step)
        f3, a3 = bin_and_average(f3, a3, fmin, fmax, step)
        
        # Apply offset correction (compensates for bin size differences)
        a = [amp - offset for amp in a]
        ax = [amp - offset for amp in ax] if ax else []
        a2 = [amp - offset for amp in a2]
        a3 = [amp - offset for amp in a3]
        
        return f, a, fx, ax, f2, a2, f3, a3
    
    
    def slice_frequency_range(freq_array, amp_array, start_f=None, end_f=None):
        """
        Trim frequency/amplitude arrays to specified range.
        """
        # Handle start
        if not start_f:  # Catches None, "", 0, etc.
            idx_min = 0
        else:
            idx_min = find_nearest(freq_array, float(start_f))
        
        # Handle end
        if not end_f:
            idx_max = len(freq_array) - 1
        else:
            idx_max = find_nearest(freq_array, float(end_f))
        
        # Ensure proper order
        if idx_min > idx_max:
            idx_min, idx_max = idx_max, idx_min
        
        return freq_array[idx_min:idx_max+1], amp_array[idx_min:idx_max+1]
    
    
    # Main processing: Process signal in frequency chunks
    # Each chunk uses different bin size (step) optimized for that frequency range
    # Offsets compensate for amplitude differences between different FFT sizes
    fout, aout, foutx, aoutx, fout2, aout2, fout3, aout3 = [], [], [], [], [], [], [], []
    
    chunk_params = [
        (20, 45, 5, 26.03),      # 20-45 Hz: 5 Hz bins
        (50, 90, 10, 19.995),    # 50-90 Hz: 10 Hz bins
        (100, 980, 20, 13.99),   # 100-980 Hz: 20 Hz bins
        (1000, 50000, 100, 0)    # 1k-50k Hz: 100 Hz bins
    ]
    
    for fmin, fmax, step, offset in chunk_params:
        f, a, fx, ax, f2, a2, f3, a3 = process_chunk(signal, Fs, fmin, fmax, step, offset)
        fout.extend(f)
        aout.extend(a)
        foutx.extend(fx)
        aoutx.extend(ax)
        fout2.extend(f2)
        aout2.extend(a2)
        fout3.extend(f3)
        aout3.extend(a3)
    
    # Apply low-pass smoothing filter to reduce noise
    sos = iirfilter(3, smoothing, btype='lowpass', output='sos')
    aout = sosfiltfilt(sos, aout)
    aout2 = sosfiltfilt(sos, aout2)
    aout3 = sosfiltfilt(sos, aout3)
    if len(aoutx) > 0:
        aoutx = sosfiltfilt(sos, aoutx)
    
    # Slice to requested frequency range
    fout, aout = slice_frequency_range(fout, aout, start_f, end_f)
    foutx, aoutx = slice_frequency_range(foutx, aoutx, start_f, end_f)
    fout2, aout2 = slice_frequency_range(fout2, aout2, start_f, end_f)
    fout3, aout3 = slice_frequency_range(fout3, aout3, start_f, end_f)
    
    return {
        'freq': fout,
        'amp': aout,
        'crosstalk_freq': foutx,
        'crosstalk_amp': aoutx,
        'harmonic2_freq': fout2,
        'harmonic2_amp': aout2,
        'harmonic3_freq': fout3,
        'harmonic3_amp': aout3,
    }
