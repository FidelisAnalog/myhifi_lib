"""
myhifi_lib - Library for audio analysis and phono cartridge measurement.

Main modules:
- sweep_extraction: Extract sweeps from phono test record measurements
- analysis: FFT-based frequency response analysis from swept sine signals
- filters: Audio filters (RIAA, STR100, XG7001)
- visualization: Plot styling and setup for frequency response plots
- constants: Test record parameters and configuration
"""

__version__ = "1.3.0"

# Sweep extraction (phono-specific)
from .sweep_extraction import extract_sweeps, find_burst_bounds, find_sweep_start, find_end_of_sweep

# Analysis (general audio DSP)
from .analysis import analyze_sweep, normalize_to_frequency, ordersignal, find_nearest, ft_window

# Filters (phono-specific and general)
from .filters import riaaiir, normxg7001, normstr100, apply_frequency_slope

# Visualization
from .visualization import (
    AUDIO_STYLE,
    DualColorHandler,
    format_freq,
    align_yaxis,
    calculate_smart_ylim,
    calculate_xlim_with_margin,
    set_ylim_tight,
    set_ylim_cascade,
    set_ylim_from_data,
    setup_rcparams,
    setup_axis_grid,
    setup_axis_ticks,
    setup_axis_xlim,
    setup_minor_tick_labels,
    setup_freq_response_axes,
    create_freq_response_figure,
    create_figure,
    create_figure_with_twin,
    plot_sweep_data,
    add_dual_channel_legend,
    add_single_channel_legend,
    add_combined_legend,
    add_delta_annotation,
    add_title,
    add_watermark,
    add_normalize_marker,
    add_footer_text,
    add_version_text,
    save_figure,
    figure_to_buffer,
    close_figure,
)

# Constants (phono-specific)
from .constants import TEST_RECORD_PARAMS

__all__ = [
    # Sweep extraction
    'extract_sweeps',
    'find_burst_bounds',
    'find_sweep_start', 
    'find_end_of_sweep',
    # Analysis
    'analyze_sweep',
    'normalize_to_frequency',
    'ordersignal',
    'find_nearest',
    'ft_window',
    # Filters
    'riaaiir',
    'normxg7001',
    'normstr100',
    'apply_frequency_slope',
    # Visualization
    'AUDIO_STYLE',
    'DualColorHandler',
    'format_freq',
    'align_yaxis',
    'calculate_smart_ylim',
    'calculate_xlim_with_margin',
    'set_ylim_tight',
    'set_ylim_cascade',
    'set_ylim_from_data',
    'setup_rcparams',
    'setup_axis_grid',
    'setup_axis_ticks',
    'setup_axis_xlim',
    'setup_minor_tick_labels',
    'setup_freq_response_axes',
    'create_freq_response_figure',
    'create_figure',
    'create_figure_with_twin',
    'plot_sweep_data',
    'add_dual_channel_legend',
    'add_single_channel_legend',
    'add_combined_legend',
    'add_delta_annotation',
    'add_title',
    'add_watermark',
    'add_normalize_marker',
    'add_footer_text',
    'add_version_text',
    'save_figure',
    'figure_to_buffer',
    'close_figure',
    # Constants
    'TEST_RECORD_PARAMS',
]
