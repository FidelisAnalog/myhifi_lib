# myhifi_lib

Python library for audio measurement and analysis, focused on phono cartridge frequency response measurement from test record sweeps.

## Modules

**analysis.py**  
FFT-based frequency analysis including `bin_and_average()` for binning scattered frequency measurements into regular intervals, and `analyze_sweep()` for extracting frequency response, crosstalk, and harmonic distortion.

**sweep_extraction.py**  
Automatic detection and extraction of frequency sweep segments from test record recordings. Handles pilot tone detection and segment slicing.

**filters.py**  
Audio correction curves including RIAA equalization, and test record compensation filters (CBS STR-100, Denon XG-7001).

**constants.py**  
Test record definitions, frequency tables, and reference data.

**visualization.py**  
Matplotlib-based plotting utilities for frequency response charts. Includes axis formatting, watermarks, legends, and multi-panel layouts.

## Installation

install from source:
```bash
pip install git+https://github.com/yourusername/myhifi_lib.git
```

## Usage

```python
from myhifi_lib import analyze_sweep, riaaiir
from myhifi_lib.sweep_extraction import extract_sweeps
from myhifi_lib.visualization import create_freq_response_figure, save_figure

# Extract sweeps from a stereo test record recording
sweeps = extract_sweeps(left_channel, right_channel, sample_rate, test_record='STR100')

# Analyze frequency response
results = analyze_sweep(sweeps['left'], sample_rate, smoothing=0.5)

# Apply RIAA de-emphasis (inverse=False) or emphasis (inverse=True)
corrected_audio = riaaiir(audio_signal, sample_rate, mode=3, inverse=False)
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## License

MIT
