"""
Sweep extraction module for phono test records.

Uses Hilbert envelope-based detection for robust, sample-rate-independent
pilot tone and sweep boundary detection.
"""

import numpy as np
from scipy.signal import sosfiltfilt, hilbert, butter
from scipy.ndimage import uniform_filter1d
import logging

from .constants import TEST_RECORD_PARAMS

logger = logging.getLogger(__name__)


def find_burst_bounds(signal, Fs, tone_freq=1000, min_duration=1.0, threshold=0.3, search_duration=30.0):
    """
    Find pilot tone burst using Hilbert envelope method.
    More robust and sample-rate independent than peak-spacing method.
    
    Parameters:
    - signal: input audio signal
    - Fs: sample rate
    - tone_freq: expected pilot tone frequency (default 1000 Hz)
    - min_duration: minimum duration in seconds for valid burst (default 1.0s)
    - threshold: normalized envelope threshold (default 0.3 = 30% of peak)
    - search_duration: duration to search in seconds (default 30s)
    
    Returns:
    - start_sample: sample index where burst starts
    - end_sample: sample index where burst ends
    """
    # Only process first search_duration seconds to keep processing fast
    search_samples = int(search_duration * Fs)
    if len(signal) > search_samples:
        logger.debug(f"Limiting search to first {search_duration}s ({search_samples} samples)")
        signal_search = signal[:search_samples]
    else:
        signal_search = signal
    
    # Bandpass filter around tone frequency (±50 Hz tolerance)
    sos = butter(4, [tone_freq - 50, tone_freq + 50], btype='band', fs=Fs, output='sos')
    filtered = sosfiltfilt(sos, signal_search)
    
    # Hilbert transform to get analytic signal and envelope
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)
    
    # Fast envelope smoothing using uniform_filter1d
    window_size = int(0.1 * Fs)
    envelope_smooth = uniform_filter1d(envelope, size=window_size, mode='nearest')
    
    # Normalize envelope
    envelope_norm = envelope_smooth / np.max(envelope_smooth)
    
    # Threshold detection
    above_threshold = envelope_norm > threshold
    
    # Find transitions (rising and falling edges)
    transitions = np.diff(above_threshold.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    # Find first sustained region above threshold
    min_samples = int(min_duration * Fs)
    
    for start, end in zip(starts, ends):
        duration = end - start
        if duration >= min_samples:
            logger.debug(f"Found burst: start={start}, end={end}, duration={duration/Fs:.2f}s")
            return start, end
    
    raise ValueError(f"No sustained pilot tone found in first {search_duration}s (min duration: {min_duration}s, threshold: {threshold})")


def find_sweep_start(signal, Fs, search_duration=20.0, threshold=0.2):
    """
    Find the start of a frequency sweep (rising energy), not a sustained tone.
    Used for test records where sweep starts several seconds after pilot tone ends.
    
    Parameters:
    - signal: raw audio signal
    - Fs: sample rate  
    - search_duration: how long to search (seconds)
    - threshold: energy rise threshold (default 0.2 = 20% of max)
    
    Returns:
    - start_sample: where sweep energy begins to rise
    """
    search_samples = int(search_duration * Fs)
    if len(signal) > search_samples:
        signal_search = signal[:search_samples]
    else:
        signal_search = signal
    
    # Bandpass around 1kHz (sweep typically starts at 1kHz)
    sos = butter(4, [900, 1100], btype='band', fs=Fs, output='sos')
    filtered = sosfiltfilt(sos, signal_search)
    
    # Get envelope
    envelope = np.abs(hilbert(filtered))
    
    # Smooth with larger window to see overall energy trend
    window_size = int(0.2 * Fs)  # 200ms window
    envelope_smooth = uniform_filter1d(envelope, size=window_size, mode='nearest')
    
    # Normalize
    envelope_norm = envelope_smooth / np.max(envelope_smooth)
    
    # Find where energy rises above threshold
    above_threshold = envelope_norm > threshold
    
    # Find first rising edge
    transitions = np.diff(above_threshold.astype(int))
    starts = np.where(transitions == 1)[0]
    
    if len(starts) > 0:
        start_sample = starts[0]
        logger.debug(f"Found sweep start at sample {start_sample} ({start_sample/Fs:.2f}s)")
        return start_sample
    else:
        raise ValueError(f"No sweep start found in first {search_duration}s")


def find_end_of_sweep(sweep_start_sample, sweep_end_min, sweep_end_max, signal, Fs, threshold=0.05):
    """
    Find end of frequency sweep using Hilbert envelope - optimized and automatic.
    Works for sweeps ending anywhere from 10kHz to 75kHz without configuration.
    
    Parameters:
    - sweep_start_sample: sample where sweep starts
    - sweep_end_min: minimum expected sweep duration (seconds)
    - sweep_end_max: maximum expected sweep duration (seconds)
    - signal: raw audio signal (not pre-filtered)
    - Fs: sample rate
    - threshold: relative amplitude threshold for end detection (default 0.05 = 5%)
    
    Returns:
    - end_sample: sample index where sweep ends
    """
    # Define search window
    sample_offset_start = sweep_start_sample + int(Fs * sweep_end_min)
    sample_offset_end = sweep_start_sample + int(Fs * sweep_end_max)
    signal_window = signal[sample_offset_start:sample_offset_end]
    
    logger.debug(f"End search window: {len(signal_window)} samples ({len(signal_window)/Fs:.2f}s)")
    
    # Use a moderate highpass (5kHz) to catch energy from sweeps ending anywhere 10kHz-75kHz
    # This is well below even the lowest sweep end, so it will catch the drop
    highpass_freq = min(5000, Fs * 0.4)  # 5kHz or 40% of Nyquist, whichever is lower
    sos = butter(4, highpass_freq, btype='high', fs=Fs, output='sos')
    filtered = sosfiltfilt(sos, signal_window)
    
    # Hilbert envelope - much cleaner than rectification
    envelope = np.abs(hilbert(filtered))
    
    # Fast smoothing with smaller window for better time resolution
    window_size = int(0.01 * Fs)  # 10ms window
    envelope_smooth = uniform_filter1d(envelope, size=window_size, mode='nearest')
    
    # Normalize
    envelope_norm = envelope_smooth / np.max(envelope_smooth)
    
    # Find where envelope drops below threshold
    below_threshold = envelope_norm < threshold
    
    # Find first sustained drop (to avoid false triggers on transients)
    min_samples = int(0.05 * Fs)  # Must stay below for 50ms
    
    # Fast vectorized method using diff to find transitions
    if len(below_threshold) >= min_samples:
        # Find transitions in/out of low region
        padded = np.concatenate(([False], below_threshold, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]  # Start of low regions
        ends = np.where(diff == -1)[0]   # End of low regions
        
        # Find first region that's >= min_samples long
        if len(starts) > 0 and len(ends) > 0:
            durations = ends - starts
            long_enough = np.where(durations >= min_samples)[0]
            
            if len(long_enough) > 0:
                end_sample = sample_offset_start + starts[long_enough[0]]
            else:
                # No sustained region, use first drop
                end_sample = sample_offset_start + starts[0]
        else:
            # No low regions at all
            end_sample = sample_offset_end
    else:
        # Window too small, use midpoint
        end_sample = (sample_offset_start + sample_offset_end) // 2
    
    logger.debug(f"End Sample (Global Index): {end_sample}")
    
    return end_sample


def extract_sweeps(left_channel, right_channel, Fs, test_record):
    """
    Extract sweep segments from stereo test record measurement.
    
    This is the main entry point for sweep extraction. It handles all the DSP logic
    for detecting pilot tones and sweep boundaries using Hilbert envelope methods.
    
    Parameters:
    - left_channel: numpy array of left channel samples
    - right_channel: numpy array of right channel samples  
    - Fs: sample rate (int)
    - test_record: string like 'TRS1007', 'STR100', etc. (case-insensitive)
    
    Returns:
    - dict with keys:
        'left': stereo array (sweep in ch0, crosstalk in ch1)
        'right': stereo array (sweep in ch0, crosstalk in ch1)
        'durations': (left_duration_sec, right_duration_sec) tuple
        'sample_rate': Fs
    
    Raises:
    - ValueError: if test_record is invalid or pilot tones not found
    """
    # Validate test record
    test_record_upper = test_record.upper()
    if test_record_upper not in TEST_RECORD_PARAMS:
        valid_records = ', '.join(sorted(TEST_RECORD_PARAMS.keys()))
        raise ValueError(f"Invalid test record '{test_record}'. Valid options: {valid_records}")
    
    params = TEST_RECORD_PARAMS[test_record_upper]
    logger.info(f"Test Record: {test_record}")
    
    # Find end of left pilot tone / start of sweep using Hilbert method
    _, start_left_sweep = find_burst_bounds(left_channel, Fs, tone_freq=1000, threshold=0.3)
    
    if params['sweep_start_detect'] == 1:
        # For test records where sweep starts several seconds after pilot ends
        # Look for energy rise at 1kHz (sweep start), not another sustained tone
        sample_offset = start_left_sweep + Fs  # Start searching 1s after pilot ends
        start_left_sweep = sample_offset + find_sweep_start(left_channel[sample_offset:], Fs, search_duration=10.0, threshold=0.2)
    
    logger.info(f"Start of Left Sweep: {start_left_sweep}")
    
    # Find end of right pilot tone / start of sweep
    sample_offset = start_left_sweep + int(Fs * params['sweep_offset'])
    _, start_right_sweep = find_burst_bounds(right_channel[sample_offset:], Fs, tone_freq=1000, threshold=0.3)
    start_right_sweep += sample_offset  # Convert to global index
    
    if params['sweep_start_detect'] == 1:
        # Same for right channel
        sample_offset = start_right_sweep + Fs
        start_right_sweep = sample_offset + find_sweep_start(right_channel[sample_offset:], Fs, search_duration=10.0, threshold=0.2)
    
    logger.info(f"Start of Right Sweep: {start_right_sweep}")
    
    # Find end of left sweep using Hilbert method
    end_left_sweep = find_end_of_sweep(
        start_left_sweep, 
        params['sweep_end_min'], 
        params['sweep_end_max'], 
        left_channel,
        Fs
    )
    logger.info(f"End of Left Sweep: {end_left_sweep}")
    
    # Find end of right sweep
    end_right_sweep = find_end_of_sweep(
        start_right_sweep, 
        params['sweep_end_min'], 
        params['sweep_end_max'], 
        right_channel,
        Fs
    )
    logger.info(f"End of Right Sweep: {end_right_sweep}")
    
    # Calculate durations
    left_duration = (end_left_sweep - start_left_sweep) / Fs
    right_duration = (end_right_sweep - start_right_sweep) / Fs
    
    logger.info(f"Left Sweep Duration: {left_duration:.2f}s")
    logger.info(f"Right Sweep Duration: {right_duration:.2f}s")
    
    # Build stereo arrays with crosstalk (sweep in ch0, crosstalk in ch1)
    left_stereo = np.column_stack((
        left_channel[start_left_sweep:end_left_sweep],
        right_channel[start_left_sweep:end_left_sweep]
    ))
    right_stereo = np.column_stack((
        right_channel[start_right_sweep:end_right_sweep],
        left_channel[start_right_sweep:end_right_sweep]
    ))
    
    return {
        'left': left_stereo,
        'right': right_stereo,
        'durations': (left_duration, right_duration),
        'sample_rate': Fs
    }
