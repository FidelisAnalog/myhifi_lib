"""
Visualization module for audio frequency response plots.

Provides consistent styling and plotting for frequency response,
harmonic distortion, and crosstalk measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import AnchoredText
from itertools import chain


# =============================================================================
# STYLE DEFINITION
# =============================================================================

AUDIO_STYLE = {
    'colors': {
        'left': '#0000ff',
        'right': '#ff0000',
        'left_h2': '#0080ff',
        'left_h3': '#00dfff',
        'right_h2': '#ff8000',
        'right_h3': '#ffdf00',
        'normalize': 'm',
    },
    'lines': {
        'fundamental': {},
        'harmonic': {'linewidth': 0.75, 'alpha': 1},
        'crosstalk': {'linestyle': (0, (3, 1, 1, 1))},
    },
    'grid': {
        'major': {'color': 'black', 'linestyle': '-'},
        'minor': {'color': 'gainsboro', 'linestyle': '-'},
    },
    'ticks': {
        'x_major': [0, 20, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000],
        'x_minor_visible': True,
        'y_minor_visible': True,
    },
    'figure': {
        'dpi': 192,
        'single': (14, 6),
        'dual_panel': (14, 10),
        'compact': (14, 3),
    },
    'annotations': {
        'title_fontsize': 16,
        'info_fontsize': 10,
        'info_alpha': 0.75,
        'delta_fontsize': 10,
    },
    'watermark': {
        'text': 'SJ',
        'color': 'm',
        'fontsize': 25,
        'alpha': 0.4,
        'style': 'oblique',
    },
}


# =============================================================================
# LEGEND HANDLER FOR DUAL-COLOR LINES
# =============================================================================

class DualColorHandler(HandlerBase):
    """
    Legend handler that draws two lines (for L/R channel representation).
    
    Used when showing both left and right channel data with matching legend entries.
    Pass tuples like ("#0000ff", "-", "#ff0000", "-") as handles.
    """
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height],
                        color=orig_handle[0], linestyle=orig_handle[1])
        l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height],
                        color=orig_handle[2], linestyle=orig_handle[3])
        return [l1, l2]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_freq(x, pos):
    """Format frequency values with k notation for tick labels."""
    if x == 0:
        return '0'
    elif x >= 1000:
        return f'{x/1000:.10g}k'.rstrip('0').rstrip('.')
    else:
        return f'{x:.10g}'


def align_yaxis(ax1, ax2):
    """
    Align zeros of two axes, preserving data ranges.
    
    Forces zero to appear on both axes, then normalizes and aligns.
    Used for dual-axis plots where fundamental and harmonics
    share a common zero reference.
    
    Returns:
        (new_lim1, new_lim2) tuples for set_ylim calls
    """
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:, 1] - y_lims[:, 0]).reshape(len(y_lims), 1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    return tuple(new_lim1.flatten()), tuple(new_lim2.flatten())


def calculate_smart_ylim(amp_data, crosstalk_data=None, has_dual=False, 
                          amp_data_2=None, crosstalk_data_2=None):
    """
    Calculate Y-axis limits based on data ranges.
    
    Uses cascading thresholds for clean axis ranges.
    """
    if has_dual and amp_data_2 is not None:
        amp_max = max(max(amp_data), max(amp_data_2))
        amp_min = min(min(amp_data), min(amp_data_2))
    else:
        amp_max = max(amp_data)
        amp_min = min(amp_data)
    
    # Check if crosstalk should determine limits
    if crosstalk_data is not None and len(crosstalk_data) > 0:
        if has_dual and crosstalk_data_2 is not None and len(crosstalk_data_2) > 0:
            xtalk_min = min(min(crosstalk_data), min(crosstalk_data_2))
        else:
            xtalk_min = min(crosstalk_data)
        return (xtalk_min - 2, amp_max + 2)
    
    # Cascading thresholds for amplitude-only
    if amp_max < 0.5:
        return (-30, 2)
    elif amp_max < 2:
        return (-29, 3)
    elif amp_max < 4:
        return (-25, 5)
    elif amp_max < 7:
        return (-25, 7)
    else:
        return (-5, 5)


def calculate_xlim_with_margin(x_data, margin_frac=0.05, min_margin_decades=0.02):
    """
    Calculate X-axis limits with log-scale margins.
    
    Parameters:
        x_data: frequency data array
        margin_frac: margin as fraction of log range (default 5%)
        min_margin_decades: minimum margin in decades
    
    Returns:
        (xmin, xmax) tuple for set_xlim
    """
    xmin = np.min(x_data)
    xmax = np.max(x_data)
    
    if xmin <= 0:
        return None  # Can't do log scale with zero/negative
    
    log_xmin = np.log10(xmin)
    log_xmax = np.log10(xmax)
    log_range = log_xmax - log_xmin
    
    margin = max(margin_frac * log_range, min_margin_decades)
    new_log_xmin = log_xmin - margin
    new_log_xmax = log_xmax + margin
    
    return (10 ** new_log_xmin, 10 ** new_log_xmax)


# =============================================================================
# Y-AXIS LIMIT FUNCTIONS
# =============================================================================

def set_ylim_tight(ax, amp_data, default=(-5, 5), override=None):
    """
    Set tight y-limits with autoscale fallback.
    
    Uses default range unless data exceeds it, then autoscales.
    Used for fundamental-only plots.
    
    Parameters:
        ax: matplotlib axes
        amp_data: amplitude data array (or list of arrays)
        default: default (min, max) range
        override: tuple to override all logic, or None
    """
    if override is not None:
        ax.set_ylim(override)
        return
    
    # Handle single array or list of arrays
    if isinstance(amp_data, np.ndarray) or (isinstance(amp_data, list) and not isinstance(amp_data[0], (list, np.ndarray))):
        data_min = min(amp_data)
        data_max = max(amp_data)
    else:
        data_min = min(min(d) for d in amp_data if len(d) > 0)
        data_max = max(max(d) for d in amp_data if len(d) > 0)
    
    ax.set_ylim(default)
    if data_min < default[0] or data_max > default[1]:
        ax.autoscale(enable=True, axis='y')


def set_ylim_cascade(ax, amp_data, override=None):
    """
    Set y-limits using cascading thresholds to maintain 5dB tick spacing.
    
    Steps through preset ranges based on data maximum.
    Used for detailed fundamental + harmonics plots.
    
    Parameters:
        ax: matplotlib axes
        amp_data: amplitude data array (or list of arrays)
        override: tuple to override all logic, or None
    """
    if override is not None:
        ax.set_ylim(override)
        return
    
    # Handle single array or list of arrays
    if isinstance(amp_data, np.ndarray) or (isinstance(amp_data, list) and not isinstance(amp_data[0], (list, np.ndarray))):
        data_max = max(amp_data)
    else:
        data_max = max(max(d) for d in amp_data if len(d) > 0)
    
    if data_max >= 7:
        ax.set_ylim(-5, 5)
    elif data_max >= 4:
        ax.set_ylim(-25, 7)
    elif data_max >= 2:
        ax.set_ylim(-25, 5)
    elif data_max >= 0.5:
        ax.set_ylim(-29, 3)
    else:
        ax.set_ylim(-30, 2)


def set_ylim_from_data(ax, amp_data, margin=2, override=None):
    """
    Set y-limits from data range with margin.
    
    Used when crosstalk or other extended-range data drives the axis.
    
    Parameters:
        ax: matplotlib axes
        amp_data: amplitude data array (or list of arrays)
        margin: margin in dB above/below data range
        override: tuple to override all logic, or None
    """
    if override is not None:
        ax.set_ylim(override)
        return
    
    # Handle single array or list of arrays
    if isinstance(amp_data, np.ndarray) or (isinstance(amp_data, list) and not isinstance(amp_data[0], (list, np.ndarray))):
        data_min = min(amp_data)
        data_max = max(amp_data)
    else:
        data_min = min(min(d) for d in amp_data if len(d) > 0)
        data_max = max(max(d) for d in amp_data if len(d) > 0)
    
    ax.set_ylim(data_min - margin, data_max + margin)


# =============================================================================
# AXIS SETUP
# =============================================================================

def setup_rcparams(style=None):
    """
    Set matplotlib rcParams for consistent tick visibility.
    Call once before creating figures.
    """
    if style is None:
        style = AUDIO_STYLE
    
    plt.rcParams['xtick.minor.visible'] = style['ticks']['x_minor_visible']
    plt.rcParams['ytick.minor.visible'] = style['ticks']['y_minor_visible']


def setup_axis_grid(ax, style=None):
    """Apply grid styling to axis."""
    if style is None:
        style = AUDIO_STYLE
    
    ax.grid(True, which='major', axis='both', **style['grid']['major'])
    ax.grid(True, which='minor', axis='both', **style['grid']['minor'])


def setup_axis_ticks(ax, style=None):
    """Set up X-axis tick locations."""
    if style is None:
        style = AUDIO_STYLE
    
    ax.set_xticks(style['ticks']['x_major'])


def setup_axis_xlim(ax, freq_data):
    """Set X-axis limits based on frequency data with smart margins."""
    xlim = calculate_xlim_with_margin(freq_data)
    if xlim:
        ax.set_xlim(xlim)


def setup_minor_tick_labels(ax):
    """
    Configure minor tick labels for narrow frequency ranges.
    
    When the axis spans less than the threshold decades,
    LogFormatter will label minor ticks - we format those too.
    """
    xlim = ax.get_xlim()
    if xlim[0] > 0:
        decades = np.log10(xlim[1] / xlim[0])
        minor_fmt = ax.xaxis.get_minor_formatter()
        if hasattr(minor_fmt, 'minor_thresholds'):
            threshold = minor_fmt.minor_thresholds[0]
            if decades < threshold:
                ax.xaxis.set_minor_formatter(FuncFormatter(format_freq))


def setup_freq_response_axes(ax, freq_data=None, y_range=None,
                              xlabel='Frequency (Hz)', ylabel='Amplitude (dB)',
                              style=None):
    """
    Full axis setup for frequency response plot.
    
    Parameters:
        ax: matplotlib axes object
        freq_data: frequency array for X limits (optional)
        y_range: (min, max) for Y limits, or None for auto
        xlabel: X-axis label
        ylabel: Y-axis label
        style: style dict
    """
    if style is None:
        style = AUDIO_STYLE
    
    setup_axis_grid(ax, style)
    setup_axis_ticks(ax, style)
    
    if freq_data is not None:
        setup_axis_xlim(ax, freq_data)
    
    if y_range is not None:
        ax.set_ylim(y_range)
    
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    setup_minor_tick_labels(ax)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_sweep_data(ax, data, channel='left', show_harmonics=True,
                    show_crosstalk=True, style=None, harmonics_ax=None):
    """
    Plot frequency response data on given axes.
    
    Parameters:
        ax: matplotlib axes for fundamental and crosstalk
        data: dict from analyze_sweep with keys:
              'freq', 'amp', 'crosstalk_freq', 'crosstalk_amp',
              'harmonic2_freq', 'harmonic2_amp', 'harmonic3_freq', 'harmonic3_amp'
        channel: 'left' or 'right' (determines colors)
        show_harmonics: whether to plot harmonic data
        show_crosstalk: whether to plot crosstalk data
        style: style dict
        harmonics_ax: separate axis for harmonics (e.g., twin axis), or None to use ax
    
    Returns:
        list of line objects created
    """
    if style is None:
        style = AUDIO_STYLE
    
    colors = style['colors']
    lines = style['lines']
    created_lines = []
    
    harm_ax = harmonics_ax if harmonics_ax is not None else ax
    
    # Fundamental
    line, = ax.semilogx(data['freq'], data['amp'],
                        color=colors[channel],
                        label='Freq Response',
                        **lines['fundamental'])
    created_lines.append(line)
    
    # 2nd Harmonic
    if show_harmonics and len(data.get('harmonic2_amp', [])) > 0:
        line, = harm_ax.semilogx(data['harmonic2_freq'], data['harmonic2_amp'],
                                  color=colors[f'{channel}_h2'],
                                  label='2ⁿᵈ Harmonic',
                                  **lines['harmonic'])
        created_lines.append(line)
    
    # 3rd Harmonic
    if show_harmonics and len(data.get('harmonic3_amp', [])) > 0:
        line, = harm_ax.semilogx(data['harmonic3_freq'], data['harmonic3_amp'],
                                  color=colors[f'{channel}_h3'],
                                  label='3ʳᵈ Harmonic',
                                  **lines['harmonic'])
        created_lines.append(line)
    
    # Crosstalk
    if show_crosstalk and len(data.get('crosstalk_amp', [])) > 0:
        line, = ax.semilogx(data['crosstalk_freq'], data['crosstalk_amp'],
                            color=colors[channel],
                            label='Crosstalk',
                            **lines['crosstalk'])
        created_lines.append(line)
    
    return created_lines


def add_dual_channel_legend(ax, has_crosstalk=True, has_harmonics=True, loc=4, style=None):
    """
    Add legend with dual-color entries for L/R channels.
    
    Parameters:
        ax: matplotlib axes
        has_crosstalk: include crosstalk entry
        has_harmonics: include harmonic entries
        loc: legend location
        style: style dict
    """
    if style is None:
        style = AUDIO_STYLE
    
    colors = style['colors']
    line_styles = style['lines']
    
    handles = [
        (colors['left'], '-', colors['right'], '-'),
    ]
    labels = ['Freq Response']
    
    if has_crosstalk:
        handles.append((colors['left'], line_styles['crosstalk']['linestyle'],
                       colors['right'], line_styles['crosstalk']['linestyle']))
        labels.append('Crosstalk')
    
    if has_harmonics:
        handles.append((colors['left_h2'], '-', colors['right_h2'], '-'))
        labels.append('2ⁿᵈ Harmonic')
        handles.append((colors['left_h3'], '-', colors['right_h3'], '-'))
        labels.append('3ʳᵈ Harmonic')
    
    ax.legend(handles, labels, handler_map={tuple: DualColorHandler()}, loc=loc)


def add_single_channel_legend(ax, loc=4):
    """Add simple legend for single channel data."""
    ax.legend(loc=loc)


def add_combined_legend(ax, ax_twin, loc=4):
    """Combine legends from main axis and twin axis."""
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc=loc)


# =============================================================================
# ANNOTATIONS
# =============================================================================

def add_delta_annotation(ax, delta_high, delta_low, freq_data, amp_data,
                         channel='left', offset_y=0, style=None):
    """
    Add +X, -Y dB annotation with styled bbox.
    
    Parameters:
        ax: matplotlib axes
        delta_high: positive deviation value
        delta_low: negative deviation value (as positive number)
        freq_data: frequency array (for positioning)
        amp_data: amplitude array (for positioning)
        channel: 'left' or 'right' (determines color)
        offset_y: additional Y offset in points (for stacking annotations)
        style: style dict
    """
    if style is None:
        style = AUDIO_STYLE
    
    color = style['colors'][channel]
    edge_color = 'b' if channel == 'left' else 'r'
    
    bbox_args = dict(boxstyle='round', color='b', fc='w', ec=edge_color, alpha=1, pad=0.15)
    
    text = f'+{delta_high}, \u2212{delta_low} dB'
    
    ax.annotate(text, color=color,
                xy=(freq_data[0], amp_data[0] - 1), xycoords='data',
                xytext=(-10, -20 + offset_y), textcoords='offset points',
                ha='left', va='center', bbox=bbox_args)


def add_title(ax, title, style=None):
    """Add title to axes."""
    if style is None:
        style = AUDIO_STYLE
    ax.set_title(title + '\n', fontsize=style['annotations']['title_fontsize'])


def add_watermark(ax, style=None):
    """Add 'SJ' watermark to axes."""
    if style is None:
        style = AUDIO_STYLE
    
    wm = style['watermark']
    anchored_text = AnchoredText(
        wm['text'],
        frameon=False, borderpad=0, pad=0.03,
        loc=1, bbox_transform=ax.transAxes,
        prop={'color': wm['color'], 'fontsize': wm['fontsize'],
              'alpha': wm['alpha'], 'style': wm['style']}
    )
    ax.add_artist(anchored_text)


def add_normalize_marker(ax, freq, mode='line', style=None):
    """
    Add normalization frequency marker.
    
    Parameters:
        ax: matplotlib axes
        freq: normalization frequency in Hz
        mode: 'line' for vertical line, 'x' for point marker
        style: style dict
    """
    if style is None:
        style = AUDIO_STYLE
    
    color = style['colors']['normalize']
    
    if mode == 'line':
        ax.axvline(x=freq, color=color, lw=1)
    else:
        ax.plot(freq, 0, marker='x', color=color, markersize=8)


def add_footer_text(fig, left_text, right_text, bottom_ax, style=None):
    """
    Add footer text below the plot.
    
    Parameters:
        fig: matplotlib figure
        left_text: text for bottom-left (equipment info)
        right_text: text for bottom-right (author)
        bottom_ax: the bottom axes object (for positioning)
        style: style dict
    """
    from matplotlib.transforms import blended_transform_factory
    
    ax_height_inches = bottom_ax.get_position().height * fig.get_figheight()
    y_in_axes_coords = -0.6 / ax_height_inches
    
    trans = blended_transform_factory(fig.transFigure, bottom_ax.transAxes)
    
    fig.text(0.125, y_in_axes_coords, left_text, alpha=.75, fontsize=10,
             transform=trans, ha='left', va='top')
    fig.text(0.9, y_in_axes_coords, right_text, alpha=.75, fontsize=10,
             transform=trans, ha='right', va='top')


def add_version_text(fig, version, filenames, web=False):
    """Add version and filename info to figure."""
    if web:
        text = f"sjplot.com/online\nSJPlot {version}\n{filenames}"
    else:
        text = f"SJPlot {version}\n{filenames}"
    
    plt.figtext(0.17, 0.118, text, fontsize=8, alpha=0.5)


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_freq_response_figure(layout='single', style=None):
    """
    Create figure with standard freq response styling.
    
    Parameters:
        layout: 'single', 'single_twin', 'dual', 'dual_twin', 'compact'
        style: style dict
    
    Returns:
        (fig, axes_dict) where axes_dict keys depend on layout:
            - 'single': {'main': ax}
            - 'single_twin': {'main': ax, 'twin': axtwin}
            - 'dual': {'top': ax0, 'bottom': ax1}
            - 'dual_twin': {'top': ax0, 'bottom': ax1, 'twin': axtwin} (twin on bottom)
            - 'compact': {'main': ax}
    """
    if style is None:
        style = AUDIO_STYLE
    
    setup_rcparams(style)
    
    # Determine figsize
    if layout in ('single', 'single_twin'):
        figsize = style['figure']['single']
    elif layout in ('dual', 'dual_twin'):
        figsize = style['figure']['dual_panel']
    elif layout == 'compact':
        figsize = style['figure']['compact']
    else:
        figsize = style['figure']['single']
    
    axes_dict = {}
    
    if layout in ('single', 'single_twin', 'compact'):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes_dict['main'] = ax
        
        if layout == 'single_twin':
            axes_dict['twin'] = ax.twinx()
            
    elif layout in ('dual', 'dual_twin'):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=figsize)
        axes_dict['top'] = ax0
        axes_dict['bottom'] = ax1
        
        if layout == 'dual_twin':
            axes_dict['twin'] = ax1.twinx()
    
    # Apply standard styling to all axes
    for key, ax in axes_dict.items():
        if key != 'twin':  # Don't grid/tick the twin axis
            setup_axis_grid(ax, style)
            setup_axis_ticks(ax, style)
            add_watermark(ax, style)
    
    return fig, axes_dict


def create_figure(layout='single', style=None):
    """
    Create figure with appropriate size for layout type.
    
    DEPRECATED: Use create_freq_response_figure() instead.
    
    Parameters:
        layout: 'single', 'dual_panel', or 'compact'
        style: style dict
    
    Returns:
        (fig, axs) tuple, axs is always a list for consistency
    """
    if style is None:
        style = AUDIO_STYLE
    
    setup_rcparams(style)
    
    figsize = style['figure'].get(layout, style['figure']['single'])
    
    if layout == 'dual_panel':
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=figsize)
        axs = list(axs)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axs = [ax]
    
    return fig, axs


def create_figure_with_twin(layout='single', style=None):
    """
    Create figure with twin Y-axis on first (or only) axes.
    
    DEPRECATED: Use create_freq_response_figure() instead.
    
    Returns:
        (fig, axs, axtwin) tuple
    """
    fig, axs = create_figure(layout, style)
    axtwin = axs[0].twinx()
    return fig, axs, axtwin


# =============================================================================
# OUTPUT
# =============================================================================

def save_figure(fig, filename, dpi=None, style=None):
    """
    Save figure with consistent settings.
    
    Parameters:
        fig: matplotlib figure
        filename: output path
        dpi: resolution, or None for style default
        style: style dict
    """
    if style is None:
        style = AUDIO_STYLE
    if dpi is None:
        dpi = style['figure']['dpi']
    
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.5, dpi=dpi)


def figure_to_buffer(fig, dpi=None, style=None):
    """
    Render figure to PNG buffer (for web use).
    
    Parameters:
        fig: matplotlib figure
        dpi: resolution, or None for style default
        style: style dict
    
    Returns:
        BytesIO buffer containing PNG data
    """
    import io
    
    if style is None:
        style = AUDIO_STYLE
    if dpi is None:
        dpi = style['figure']['dpi']
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5, dpi=dpi)
    buf.seek(0)
    return buf


def close_figure(fig):
    """Close figure to free memory."""
    plt.close(fig)
