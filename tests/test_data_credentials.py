"""Tests for WRDS credential validation."""

import os
import sys
from enum import Enum
from io import StringIO
from unittest.mock import patch

import pytest


class CredentialStatus(Enum):
    """Enum for credential validation status."""

    VALID = "valid"
    MISSING_USERNAME = "missing_username"
    MISSING_PASSWORD = "missing_password"
    BOTH_MISSING = "both_missing"


# Import functions to test (these will be created later)
def test_check_wrds_credentials_valid():
    """Test credentials are valid when both env vars are set."""
    from src.data.credentials import check_wrds_credentials

    with patch.dict(os.environ, {"WRDS_USERNAME": "test_user", "WRDS_PASSWORD": "test_pass"}):
        result = check_wrds_credentials()
        assert result == CredentialStatus.VALID.value


def test_check_wrds_credentials_missing_username():
    """Test missing username detection."""
    from src.data.credentials import check_wrds_credentials

    with patch.dict(os.environ, {}, clear=True):
        with patch.dict(os.environ, {"WRDS_PASSWORD": "test_pass"}):
            result = check_wrds_credentials()
            assert result == CredentialStatus.MISSING_USERNAME.value


def test_check_wrds_credentials_missing_password():
    """Test missing password detection."""
    from src.data.credentials import check_wrds_credentials

    with patch.dict(os.environ, {}, clear=True):
        with patch.dict(os.environ, {"WRDS_USERNAME": "test_user"}):
            result = check_wrds_credentials()
            assert result == CredentialStatus.MISSING_PASSWORD.value


def test_check_wrds_credentials_both_missing():
    """Test both missing detection."""
    from src.data.credentials import check_wrds_credentials

    with patch.dict(os.environ, {}, clear=True):
        result = check_wrds_credentials()
        assert result == CredentialStatus.BOTH_MISSING.value


def test_validate_and_exit_exits_when_missing():
    """Test validate_and_exit exits with code 1 when credentials missing."""
    from src.data.credentials import validate_and_exit

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(sys, "exit") as mock_exit:
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                validate_and_exit()
                mock_exit.assert_called_once_with(1)
                output = mock_stdout.getvalue()
                assert "ERROR: WRDS credentials not found" in output
                assert "WRDS_USERNAME" in output
                assert "WRDS_PASSWORD" in output
