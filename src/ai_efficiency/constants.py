"""
Constants for AI efficiency calculations.
"""

# Carbon intensity by region (g CO2 per kWh)
# Source: IEA, Electricity Maps, regional grid operators
CARBON_INTENSITY = {
    # Asia
    "KR": 450,      # South Korea
    "JP": 470,      # Japan
    "CN": 580,      # China
    "IN": 710,      # India
    "SG": 400,      # Singapore

    # Europe
    "EU": 250,      # EU average
    "DE": 350,      # Germany
    "FR": 50,       # France (nuclear)
    "UK": 200,      # United Kingdom
    "NO": 20,       # Norway (hydro)
    "PL": 650,      # Poland (coal)

    # North America
    "US": 380,      # US average
    "US-CA": 200,   # California
    "US-TX": 400,   # Texas
    "US-WA": 80,    # Washington (hydro)
    "CA": 120,      # Canada

    # Other
    "AU": 550,      # Australia
    "BR": 80,       # Brazil (hydro)
    "WORLD": 450,   # World average
}

# Efficiency grade thresholds
GRADES = {
    "A+": 100_000,
    "A": 50_000,
    "B": 10_000,
    "C": 1_000,
    "D": 0,
}

# Hardware power consumption estimates (Watts)
HARDWARE_POWER = {
    # NVIDIA GPUs
    "A100": 400,
    "H100": 700,
    "V100": 300,
    "RTX4090": 450,
    "RTX3090": 350,
    "RTX3080": 320,
    "T4": 70,

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

    # CPU (average)
    "CPU": 65,
    "CPU-server": 150,

    # TPU
    "TPUv4": 170,
    "TPUv5": 200,
}

# Default assumptions
DEFAULT_REGION = "WORLD"
DEFAULT_HARDWARE = "CPU"
