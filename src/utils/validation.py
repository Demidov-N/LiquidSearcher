"""Validation utilities for the Liquidity Risk Management project."""

from dataclasses import dataclass
from datetime import datetime

import polars as pl

OHLCV_COLUMNS: set[str] = {"date", "symbol", "open", "high", "low", "close", "volume"}
FUNDAMENTAL_COLUMNS: set[str] = {
    "date",
    "symbol",
    "market_cap",
    "book_value",
    "sales",
    "net_income",
    "debt",
    "cash",
}


@dataclass
class DataFrameSchema:
    """Schema definition for DataFrame validation."""

    required_columns: set[str]
    optional_columns: set[str] | None = None

    def validate(self, df: pl.DataFrame) -> list[str]:
        """Validate DataFrame against schema.

        Args:
            df: DataFrame to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        df_columns = set(df.columns)

        missing = self.required_columns - df_columns
        if missing:
            errors.append(f"Missing required columns: {missing}")

        return errors


def validate_ohlcv_columns(df: pl.DataFrame) -> list[str]:
    """Validate OHLCV DataFrame has required columns.

    Args:
        df: DataFrame to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    schema = DataFrameSchema(required_columns=OHLCV_COLUMNS)
    return schema.validate(df)


def validate_fundamental_columns(df: pl.DataFrame) -> list[str]:
    """Validate fundamental data DataFrame has required columns.

    Args:
        df: DataFrame to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    schema = DataFrameSchema(required_columns=FUNDAMENTAL_COLUMNS)
    return schema.validate(df)


def validate_date_range(
    df: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    date_column: str = "date",
) -> list[str]:
    """Validate DataFrame date range.

    Args:
        df: DataFrame to validate.
        start_date: Minimum date (YYYY-MM-DD format).
        end_date: Maximum date (YYYY-MM-DD format).
        date_column: Name of date column.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    if date_column not in df.columns:
        errors.append(f"Date column '{date_column}' not found")
        return errors

    if start_date is not None:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            min_date = df.select(date_column).to_series().min()
            if min_date is not None and min_date < start_dt:
                errors.append(f"Data contains dates before {start_date}")
        except ValueError:
            errors.append(f"Invalid start_date format: {start_date}")

    if end_date is not None:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            max_date = df.select(date_column).to_series().max()
            if max_date is not None and max_date > end_dt:
                errors.append(f"Data contains dates after {end_date}")
        except ValueError:
            errors.append(f"Invalid end_date format: {end_date}")

    return errors


def validate_no_nulls(
    df: pl.DataFrame,
    columns: list[str] | None = None,
) -> list[str]:
    """Validate specified columns have no null values.

    Args:
        df: DataFrame to validate.
        columns: Columns to check (default: all columns).

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    cols = columns or df.columns

    for col in cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found")
            continue

        null_count = df[col].null_count()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null values")

    return errors


def validate_positive_values(
    df: pl.DataFrame,
    columns: list[str],
) -> list[str]:
    """Validate specified columns contain only positive values.

    Args:
        df: DataFrame to validate.
        columns: Columns that must be positive.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    for col in columns:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found")
            continue

        if df[col].dtype == pl.Float64 or df[col].dtype == pl.Int64:
            min_val = df[col].min()
            if min_val is not None and min_val <= 0:
                errors.append(f"Column '{col}' contains non-positive values")

    return errors
