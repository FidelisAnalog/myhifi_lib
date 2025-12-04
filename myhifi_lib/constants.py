"""
Constants and configuration parameters for myhifi_lib.
"""

# Test record parameters for sweep extraction
# Each record has different timing characteristics for pilot tones and sweeps
TEST_RECORD_PARAMS = {
    'TRS1007': {
        'sweep_offset': 74,      # Seconds between left and right pilot tones
        'sweep_end_min': 48,     # Minimum expected sweep duration (seconds)
        'sweep_end_max': 52,     # Maximum expected sweep duration (seconds)
        'sweep_start_detect': 0  # 0=sweep starts at pilot end, 1=sweep starts seconds after pilot
    },
    'TRS1005': {
        'sweep_offset': 32,
        'sweep_end_min': 26,
        'sweep_end_max': 34,
        'sweep_start_detect': 1
    },
    'STR100': {
        'sweep_offset': 74,
        'sweep_end_min': 63,
        'sweep_end_max': 67,
        'sweep_start_detect': 0
    },
    'STR120': {
        'sweep_offset': 58,
        'sweep_end_min': 45,
        'sweep_end_max': 50,
        'sweep_start_detect': 0
    },
    'STR130': {
        'sweep_offset': 82,
        'sweep_end_min': 63,
        'sweep_end_max': 67,
        'sweep_start_detect': 0
    },
    'STR170': {
        'sweep_offset': 75,
        'sweep_end_min': 63,
        'sweep_end_max': 67,
        'sweep_start_detect': 0
    },
    'QR2009': {
        'sweep_offset': 80,
        'sweep_end_min': 48,
        'sweep_end_max': 52,
        'sweep_start_detect': 0
    },
    'QR2010': {
        'sweep_offset': 24,
        'sweep_end_min': 15,
        'sweep_end_max': 18,
        'sweep_start_detect': 0
    },
    'XG7001': {
        'sweep_offset': 78,
        'sweep_end_min': 48,
        'sweep_end_max': 52,
        'sweep_start_detect': 0
    },
    'XG7002': {
        'sweep_offset': 65,
        'sweep_end_min': 26,
        'sweep_end_max': 30,
        'sweep_start_detect': 1
    },
    'XG7005': {
        'sweep_offset': 78,
        'sweep_end_min': 48,
        'sweep_end_max': 52,
        'sweep_start_detect': 0
    },
    'DIN45543': {
        'sweep_offset': 78,
        'sweep_end_min': 48,
        'sweep_end_max': 52,
        'sweep_start_detect': 0
    },
    'ИЗМ33С0327': {
        'sweep_offset': 58,
        'sweep_end_min': 48,
        'sweep_end_max': 52,
        'sweep_start_detect': 0
    },
}
