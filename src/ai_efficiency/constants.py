"""
Constants for AI efficiency calculations.
"""

# Carbon intensity by region (g CO2 per kWh)
# Source: Electricity Maps snapshot 2026-01-15
# Reference: GES Paper (NeurIPS 2026 D&B) - 15 countries analyzed
CARBON_INTENSITY = {
    # Low carbon (< 100 gCO2/kWh)
    "NO": 20,       # Norway (hydro)
    "SE": 45,       # Sweden
    "FR": 50,       # France (nuclear)
    "BR": 80,       # Brazil (hydro)

    # Medium-low (100-250 gCO2/kWh)
    "CA": 150,      # Canada
    "US-CA": 200,   # California
    "UK": 230,      # United Kingdom
    "EU": 250,      # EU average

    # Medium (250-450 gCO2/kWh)
    "DE": 380,      # Germany
    "US": 400,      # US average
    "JP": 450,      # Japan
    "KR": 450,      # South Korea

    # High carbon (> 450 gCO2/kWh)
    "AU": 500,      # Australia
    "CN": 550,      # China
    "PL": 650,      # Poland (coal)
    "IN": 700,      # India

    # Other regions
    "SG": 400,      # Singapore
    "US-TX": 400,   # Texas
    "US-WA": 80,    # Washington (hydro)
    "WORLD": 450,   # World average
}

# Efficiency grade thresholds (derived from 74-model benchmark population percentiles)
# A+ = Top 10%, A = Top 25%, B = Top 50%, C = Top 75%, D = Bottom 25%
# Reference: GES Paper (NeurIPS 2026 D&B)
GRADES = {
    "A+": 3_265_200,   # Top 10% (P90)
    "A": 1_306_469,    # Top 25% (P75)
    "B": 512_892,      # Top 50% (P50 = Median)
    "C": 187_135,      # Top 75% (P25)
    "D": 0,            # Bottom 25%
}

# Legacy thresholds (deprecated, kept for backwards compatibility)
GRADES_LEGACY = {
    "A+": 100_000,
    "A": 50_000,
    "B": 10_000,
    "C": 1_000,
    "D": 0,
}

# Hardware power consumption estimates (Watts)
# Reference: GES Paper (NeurIPS 2026 D&B) - 16 hardware platforms benchmarked
HARDWARE_POWER = {
    # NVIDIA Datacenter GPUs
    "A100": 400,
    "H100": 700,
    "V100": 300,
    "T4": 70,

    # NVIDIA Consumer GPUs
    "RTX4090": 450,
    "RTX3090": 350,
    "RTX3080": 320,

    # AMD GPUs
    "MI300X": 750,
    "MI250X": 560,

    # Apple Silicon
    "M1": 20,
    "M2": 22,
    "M3": 25,
    "M1-Pro": 30,
    "M1-Max": 40,
    "M2-Ultra": 60,

    # Edge GPUs (NVIDIA Jetson)
    "Jetson-Nano": 10,
    "Jetson-Xavier-NX": 15,
    "Jetson-Orin-Nano": 15,

    # Mobile NPUs
    "Snapdragon-8-Gen3": 5,
    "MediaTek-Dimensity-9300": 4,

    # Edge TPU
    "Coral-TPU": 2,

    # ARM CPU (Raspberry Pi)
    "RPi-4B": 6,
    "RPi-5": 8,

    # CPU (general)
    "CPU": 65,
    "CPU-server": 150,
    "CPU-ARM": 8,

    # TPU (Cloud)
    "TPUv4": 170,
    "TPUv5": 200,
}

# Default assumptions
DEFAULT_REGION = "WORLD"
DEFAULT_HARDWARE = "CPU"
