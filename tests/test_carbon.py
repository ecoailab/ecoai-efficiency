"""Tests for carbon intensity module."""

import pytest


def test_get_carbon_intensity_static():
    """Test static carbon intensity lookup."""
    from ai_efficiency import get_carbon_intensity

    # Test known regions
    intensity = get_carbon_intensity("KR", real_time=False)
    assert intensity == 450

    intensity = get_carbon_intensity("FR", real_time=False)
    assert intensity == 50

    intensity = get_carbon_intensity("NO", real_time=False)
    assert intensity == 20


def test_get_carbon_intensity_fallback():
    """Test fallback to world average for unknown region."""
    from ai_efficiency import get_carbon_intensity

    intensity = get_carbon_intensity("XX", real_time=False)
    assert intensity == 450  # WORLD average


def test_get_carbon_data():
    """Test get_carbon_data returns full data object."""
    from ai_efficiency import get_carbon_data

    data = get_carbon_data("KR", real_time=False)

    assert data.intensity == 450
    assert data.zone == "KR"
    assert data.source == "fallback"


def test_carbon_intensity_provider():
    """Test CarbonIntensityProvider class."""
    from ai_efficiency.carbon import CarbonIntensityProvider

    provider = CarbonIntensityProvider()

    # Should work without API key (fallback mode)
    data = provider.get_intensity("US", real_time=False)
    assert data.intensity == 380
    assert data.source == "fallback"


def test_cloud_carbon_factors():
    """Test cloud provider carbon factors."""
    from ai_efficiency import get_cloud_carbon_factor

    # AWS
    aws_east = get_cloud_carbon_factor("aws", "us-east-1")
    assert aws_east == 379

    # GCP
    gcp_west = get_cloud_carbon_factor("gcp", "us-west1")
    assert gcp_west == 92  # Low carbon (hydro/renewables)

    # Azure
    azure_korea = get_cloud_carbon_factor("azure", "koreacentral")
    assert azure_korea == 500


def test_cloud_carbon_unknown_provider():
    """Test unknown cloud provider returns world average."""
    from ai_efficiency import get_cloud_carbon_factor

    with pytest.warns(UserWarning, match="Unknown cloud provider"):
        intensity = get_cloud_carbon_factor("unknown", "region")
        assert intensity == 450


def test_carbon_data_str():
    """Test CarbonIntensityData string representation."""
    from ai_efficiency import get_carbon_data

    data = get_carbon_data("DE", real_time=False)
    output = str(data)

    assert "350" in output  # Germany intensity
    assert "DE" in output
    assert "fallback" in output
