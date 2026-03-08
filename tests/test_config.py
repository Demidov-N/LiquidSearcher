"""Test configuration system."""

from src.config.settings import Settings, get_settings


def test_settings_singleton():
    """Test that settings is a singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_default_values():
    """Test default configuration values."""
    s = Settings()

    assert s.data.train_start == "2010-01-01"
    assert s.data.train_end == "2022-12-31"

    assert s.features.beta_window == 60
    assert s.features.winsorize_lower == 0.01

    assert s.model.joint_dim == 256
    assert s.model.temperature == 0.07

    assert s.liquidity.spread_bps_threshold == 50.0
