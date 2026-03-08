"""Tests for WRDS connection module."""

import pytest

from src.data import WRDSConfig, WRDSConnection


class TestWRDSConfig:
    def test_wrds_config_creation(self):
        """Test WRDSConfig dataclass creation with default values."""
        config = WRDSConfig()
        assert config.username is None
        assert config.password is None
        assert config.host == "wrds-cloud.wharton.upenn.edu"
        assert config.port == 9737
        assert config.mock_mode is False

    def test_wrds_config_with_custom_values(self):
        """Test WRDSConfig with custom credentials."""
        config = WRDSConfig(
            username="testuser",
            password="testpass",
            host="custom.host.com",
            port=1234,
            mock_mode=True,
        )
        assert config.username == "testuser"
        assert config.password == "testpass"
        assert config.host == "custom.host.com"
        assert config.port == 1234
        assert config.mock_mode is True

    def test_wrds_config_get_username(self):
        """Test get_username returns config value."""
        config = WRDSConfig(username="myuser")
        assert config.get_username() == "myuser"

    def test_wrds_config_get_password(self):
        """Test get_password returns config value."""
        config = WRDSConfig(password="mypass")
        assert config.get_password() == "mypass"

    def test_wrds_config_has_credentials(self):
        """Test has_credentials detection."""
        config_no_creds = WRDSConfig()
        assert config_no_creds.has_credentials() is False

        config_with_creds = WRDSConfig(username="user", password="pass")
        assert config_with_creds.has_credentials() is True


class TestWRDSConnection:
    def test_wrds_connection_context_manager(self):
        """Test WRDSConnection as context manager."""
        config = WRDSConfig(mock_mode=True)
        with WRDSConnection(config) as conn:
            assert conn.is_connected() is True

    def test_wrds_connection_mock_mode(self):
        """Test connection enters mock mode without credentials."""
        config = WRDSConfig()
        conn = WRDSConnection(config)
        conn.connect()
        assert conn.is_mock_mode() is True
        assert conn.is_connected() is True
        conn.disconnect()
        assert conn.is_connected() is False

    def test_wrds_connection_disconnect(self):
        """Test disconnect closes connection."""
        config = WRDSConfig(mock_mode=True)
        conn = WRDSConnection(config)
        conn.connect()
        assert conn.is_connected() is True
        conn.disconnect()
        assert conn.is_connected() is False

    def test_wrds_connection_get_connection_in_mock_mode(self):
        """Test get_connection returns None in mock mode."""
        config = WRDSConfig(mock_mode=True)
        conn = WRDSConnection(config)
        conn.connect()
        assert conn.get_connection() is None
        conn.disconnect()
