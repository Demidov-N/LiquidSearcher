"""WRDS credentials validation module.

Provides fail-fast credential checking with clear error messages.
"""

import os
import sys
from enum import Enum


class CredentialStatus(Enum):
    """Enum for credential validation status."""

    VALID = "valid"
    MISSING_USERNAME = "missing_username"
    MISSING_PASSWORD = "missing_password"
    BOTH_MISSING = "both_missing"


def check_wrds_credentials() -> str:
    """Check WRDS credential environment variables.

    Returns:
        One of CredentialStatus values:
        - "valid": Both WRDS_USERNAME and WRDS_PASSWORD are set
        - "missing_username": Only WRDS_PASSWORD is set
        - "missing_password": Only WRDS_USERNAME is set
        - "both_missing": Neither credential is set
    """
    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")

    has_username = bool(username)
    has_password = bool(password)

    if has_username and has_password:
        return CredentialStatus.VALID.value
    elif not has_username and not has_password:
        return CredentialStatus.BOTH_MISSING.value
    elif not has_username:
        return CredentialStatus.MISSING_USERNAME.value
    else:
        return CredentialStatus.MISSING_PASSWORD.value


def validate_and_exit(use_mock: bool = False) -> None:
    """Validate credentials and exit with helpful message if missing.

    Args:
        use_mock: If True, skip credential validation (for testing with mock data)

    Exits with code 1 if credentials are missing and use_mock is False.
    """
    if use_mock:
        return

    status = check_wrds_credentials()

    if status != CredentialStatus.VALID.value:
        print()
        print(
            "❌ ERROR: WRDS credentials not found. Set WRDS_USERNAME and WRDS_PASSWORD environment variables."
        )
        print()
        print("To set credentials:")
        print("  export WRDS_USERNAME=your_username")
        print("  export WRDS_PASSWORD=your_password")
        print()
        print("Or run with mock data explicitly (for testing only):")
        print("  python -m scripts.preprocess_features --use-mock")
        print()
        sys.exit(1)
