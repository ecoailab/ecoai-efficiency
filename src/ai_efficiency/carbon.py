"""
Real-time carbon intensity integration.

Fetches live grid carbon intensity data from Electricity Maps API.
Falls back to static regional averages when API is unavailable.

References:
- Electricity Maps: https://www.electricitymaps.com/
- Cloud Carbon Footprint: https://www.cloudcarbonfootprint.org/
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict
from functools import lru_cache
import warnings

from .constants import CARBON_INTENSITY

# Cache TTL in seconds (15 minutes)
CACHE_TTL = 900

# Electricity Maps zone codes
ZONE_MAPPING = {
    # Asia
    "KR": "KR",
    "JP": "JP",
    "CN": "CN",
    "IN": "IN-WE",
    "SG": "SG",
    # Europe
    "EU": "DE",  # Use Germany as EU proxy
    "DE": "DE",
    "FR": "FR",
    "UK": "GB",
    "NO": "NO-NO1",
    "PL": "PL",
    # North America
    "US": "US-CAL-CISO",  # California as default
    "US-CA": "US-CAL-CISO",
    "US-TX": "US-TEX-ERCO",
    "US-WA": "US-NW-BPAT",
    "CA": "CA-ON",
    # Other
    "AU": "AU-NSW",
    "BR": "BR-CS",
}


@dataclass
class CarbonIntensityData:
    """Real-time carbon intensity data."""

    intensity: float  # gCO2eq/kWh
    zone: str
    source: str  # "api" or "fallback"
    timestamp: Optional[str] = None
    is_cached: bool = False

    def __str__(self) -> str:
        source_str = f"({self.source})"
        if self.is_cached:
            source_str += " [cached]"
        return f"{self.intensity:.0f} gCO2/kWh in {self.zone} {source_str}"


class CarbonIntensityProvider:
    """
    Provider for carbon intensity data.

    Uses Electricity Maps API when available, falls back to static values.

    Usage:
        >>> provider = CarbonIntensityProvider(api_key="your-api-key")
        >>> data = provider.get_intensity("KR")
        >>> print(data.intensity)  # gCO2eq/kWh

    Without API key:
        >>> provider = CarbonIntensityProvider()
        >>> data = provider.get_intensity("KR")  # Uses static fallback
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the carbon intensity provider.

        Args:
            api_key: Electricity Maps API key (optional).
                    Can also be set via ELECTRICITY_MAPS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ELECTRICITY_MAPS_API_KEY")
        self._cache: Dict[str, tuple] = {}  # zone -> (data, timestamp)

    def get_intensity(self, region: str, real_time: bool = True) -> CarbonIntensityData:
        """
        Get carbon intensity for a region.

        Args:
            region: Region code (e.g., "KR", "US", "EU")
            real_time: If True and API key available, fetch live data

        Returns:
            CarbonIntensityData with intensity and metadata
        """
        # Check cache first
        if region in self._cache:
            cached_data, cached_time = self._cache[region]
            if time.time() - cached_time < CACHE_TTL:
                cached_data.is_cached = True
                return cached_data

        # Try real-time API
        if real_time and self.api_key:
            try:
                data = self._fetch_from_api(region)
                if data:
                    self._cache[region] = (data, time.time())
                    return data
            except Exception as e:
                warnings.warn(f"Failed to fetch real-time carbon data: {e}")

        # Fallback to static values
        return self._get_fallback(region)

    def _fetch_from_api(self, region: str) -> Optional[CarbonIntensityData]:
        """Fetch real-time data from Electricity Maps API."""
        try:
            import urllib.request
            import json

            zone = ZONE_MAPPING.get(region, region)
            url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"

            req = urllib.request.Request(url)
            req.add_header("auth-token", self.api_key)

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

                return CarbonIntensityData(
                    intensity=data["carbonIntensity"],
                    zone=zone,
                    source="api",
                    timestamp=data.get("datetime"),
                )
        except Exception:
            return None

    def _get_fallback(self, region: str) -> CarbonIntensityData:
        """Get static fallback carbon intensity."""
        intensity = CARBON_INTENSITY.get(region, CARBON_INTENSITY["WORLD"])
        return CarbonIntensityData(
            intensity=intensity,
            zone=region,
            source="fallback",
        )


# Global provider instance
_provider: Optional[CarbonIntensityProvider] = None


def get_carbon_intensity(
    region: str,
    real_time: bool = True,
    api_key: Optional[str] = None,
) -> float:
    """
    Get carbon intensity for a region (convenience function).

    Args:
        region: Region code (e.g., "KR", "US", "EU")
        real_time: If True and API key available, fetch live data
        api_key: Optional Electricity Maps API key

    Returns:
        Carbon intensity in gCO2eq/kWh

    Example:
        >>> from ai_efficiency.carbon import get_carbon_intensity
        >>> intensity = get_carbon_intensity("KR", real_time=True)
        >>> print(f"Current grid intensity: {intensity} gCO2/kWh")
    """
    global _provider

    if _provider is None or (api_key and _provider.api_key != api_key):
        _provider = CarbonIntensityProvider(api_key=api_key)

    data = _provider.get_intensity(region, real_time=real_time)
    return data.intensity


def get_carbon_data(
    region: str,
    real_time: bool = True,
    api_key: Optional[str] = None,
) -> CarbonIntensityData:
    """
    Get full carbon intensity data with metadata.

    Args:
        region: Region code (e.g., "KR", "US", "EU")
        real_time: If True and API key available, fetch live data
        api_key: Optional Electricity Maps API key

    Returns:
        CarbonIntensityData with intensity, source, and timestamp

    Example:
        >>> from ai_efficiency.carbon import get_carbon_data
        >>> data = get_carbon_data("FR")
        >>> print(data)  # "50 gCO2/kWh in FR (fallback)"
        >>> print(f"Source: {data.source}")
    """
    global _provider

    if _provider is None or (api_key and _provider.api_key != api_key):
        _provider = CarbonIntensityProvider(api_key=api_key)

    return _provider.get_intensity(region, real_time=real_time)


# Cloud provider carbon factors (marginal emissions, gCO2eq/kWh)
# Source: Cloud Carbon Footprint project
CLOUD_CARBON_FACTORS = {
    "aws": {
        "us-east-1": 379,
        "us-east-2": 440,
        "us-west-1": 322,
        "us-west-2": 322,
        "eu-west-1": 316,
        "eu-west-2": 225,
        "eu-central-1": 338,
        "ap-northeast-1": 506,  # Tokyo
        "ap-northeast-2": 500,  # Seoul
        "ap-southeast-1": 493,  # Singapore
    },
    "gcp": {
        "us-central1": 394,
        "us-east1": 379,
        "us-west1": 92,
        "europe-west1": 135,
        "europe-west4": 164,
        "asia-east1": 541,
        "asia-northeast1": 506,
        "asia-northeast3": 500,  # Seoul
    },
    "azure": {
        "eastus": 379,
        "eastus2": 440,
        "westus": 322,
        "westus2": 322,
        "westeurope": 316,
        "northeurope": 316,
        "japaneast": 506,
        "koreacentral": 500,
    },
}


def get_cloud_carbon_factor(provider: str, region: str) -> float:
    """
    Get carbon intensity for a cloud provider region.

    Args:
        provider: Cloud provider ("aws", "gcp", "azure")
        region: Provider-specific region code

    Returns:
        Carbon intensity in gCO2eq/kWh

    Example:
        >>> from ai_efficiency.carbon import get_cloud_carbon_factor
        >>> intensity = get_cloud_carbon_factor("aws", "us-east-1")
        >>> print(f"AWS us-east-1: {intensity} gCO2/kWh")
    """
    provider = provider.lower()

    if provider not in CLOUD_CARBON_FACTORS:
        warnings.warn(f"Unknown cloud provider: {provider}. Using world average.")
        return CARBON_INTENSITY["WORLD"]

    regions = CLOUD_CARBON_FACTORS[provider]

    if region not in regions:
        warnings.warn(f"Unknown region {region} for {provider}. Using world average.")
        return CARBON_INTENSITY["WORLD"]

    return regions[region]
