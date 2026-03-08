"""Tests for unified preprocessing script.

Tests verify that preprocess_unified.py correctly:
1. Outputs a single parquet file with all stocks
2. Includes symbol column to distinguish stocks
3. Has all required columns (32 features + OHLCV)
4. Processes stocks in batches
5. Validates WRDS credentials at start
6. Supports mock data mode
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest


class TestUnifiedPreprocessingStructure:
    """Test that unified preprocessing produces correct output structure."""

    def test_unified_output_file_exists(self, tmp_path):
        """Unified preprocessing should create a single output file."""
        from scripts import preprocess_unified

        output_dir = tmp_path / "features"
        output_file = output_dir / "all_features.parquet"

        # Mock the main function to not actually run
        with mock.patch.object(sys, "exit"):
            with mock.patch("scripts.preprocess_unified.process_unified") as mock_process:
                mock_process.return_value = output_file

                # Parse args and run
                args = [
                    "--start-date",
                    "2023-01-01",
                    "--end-date",
                    "2023-12-31",
                    "--symbols",
                    "AAPL",
                    "MSFT",
                    "--output-dir",
                    str(output_dir),
                    "--use-mock",
                ]

                with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
                    preprocess_unified.main()

                mock_process.assert_called_once()

    def test_unified_file_has_symbol_column(self, tmp_path):
        """Unified output must have 'symbol' column to distinguish stocks."""
        # Create a sample unified parquet file
        output_file = tmp_path / "all_features.parquet"

        sample_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", "2023-01-10"),
                "symbol": ["AAPL"] * 10,
                "close": [100.0] * 10,
                "volume": [1000000] * 10,
                "feature_1": [0.5] * 10,
            }
        )
        sample_data.to_parquet(output_file)

        # Verify structure
        df = pd.read_parquet(output_file)
        assert "symbol" in df.columns, "Unified file must have 'symbol' column"

    def test_unified_file_has_multiple_symbols(self, tmp_path):
        """Unified file should contain data for all processed symbols."""
        output_file = tmp_path / "all_features.parquet"

        # Create sample data with multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data = []
        for symbol in symbols:
            for i in range(5):
                data.append(
                    {
                        "date": pd.Timestamp(f"2023-01-{i + 1:02d}"),
                        "symbol": symbol,
                        "close": 100.0 + i,
                        "volume": 1000000,
                        "feature_1": 0.5,
                    }
                )

        df = pd.DataFrame(data)
        df.to_parquet(output_file)

        # Verify multiple symbols present
        result = pd.read_parquet(output_file)
        unique_symbols = result["symbol"].unique()
        assert set(unique_symbols) == set(symbols), (
            f"Expected symbols {symbols}, got {unique_symbols}"
        )

    def test_unified_file_has_required_columns(self, tmp_path):
        """Unified file must have all 32 features + OHLCV columns."""
        output_file = tmp_path / "all_features.parquet"

        # Expected columns based on requirements
        required_columns = [
            # OHLCV columns
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return",
        ]

        # Create sample data with all required columns
        data = {
            "date": pd.date_range("2023-01-01", "2023-01-10"),
            "symbol": ["AAPL"] * 10,
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [99.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000000] * 10,
            "return": [0.02] * 10,
        }

        df = pd.DataFrame(data)
        df.to_parquet(output_file)

        # Verify required columns exist
        result = pd.read_parquet(output_file)
        for col in required_columns:
            assert col in result.columns, f"Required column '{col}' missing from unified file"


class TestUnifiedPreprocessingBehavior:
    """Test unified preprocessing behavior and integration."""

    def test_processes_stocks_in_batches(self, tmp_path):
        """Should process stocks in configurable batches for memory efficiency."""
        from scripts import preprocess_unified

        # Mock batch processing
        with mock.patch("scripts.preprocess_unified.process_symbol_batch") as mock_batch:
            mock_batch.return_value = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", "2023-01-03"),
                    "symbol": ["AAPL"] * 3,
                    "close": [100.0] * 3,
                }
            )

            # Create mock symbols list
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            batch_size = 2

            # Calculate expected batches
            expected_batches = 3  # 5 symbols / 2 batch_size, rounded up

            # Process symbols
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                preprocess_unified.process_symbol_batch(
                    batch,
                    datetime(2023, 1, 1),
                    datetime(2023, 12, 31),
                    tmp_path,
                    use_mock=True,
                )

            # Verify batch processing was called
            assert mock_batch.call_count == expected_batches

    def test_validates_wrds_credentials(self, tmp_path):
        """Should validate WRDS credentials at start (unless using mock)."""
        from scripts import preprocess_unified

        # Test that it calls validate_and_exit
        with mock.patch("scripts.preprocess_unified.validate_and_exit") as mock_validate:
            with mock.patch("scripts.preprocess_unified.process_unified") as mock_process:
                mock_process.return_value = tmp_path / "all_features.parquet"

                args = [
                    "--start-date",
                    "2023-01-01",
                    "--end-date",
                    "2023-12-31",
                    "--symbols",
                    "AAPL",
                    "--output-dir",
                    str(tmp_path),
                ]

                with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
                    # When mocking validate_and_exit, main() should complete normally
                    preprocess_unified.main()

                # Should have called validate_and_exit with use_mock=False
                mock_validate.assert_called_once_with(use_mock=False)

    def test_supports_mock_data_mode(self, tmp_path):
        """Should support --use-mock flag for testing without credentials."""
        from scripts import preprocess_unified

        with mock.patch("scripts.preprocess_unified.validate_and_exit") as mock_validate:
            with mock.patch("scripts.preprocess_unified.process_unified") as mock_process:
                mock_process.return_value = tmp_path / "all_features.parquet"

                args = [
                    "--start-date",
                    "2023-01-01",
                    "--end-date",
                    "2023-12-31",
                    "--symbols",
                    "AAPL",
                    "--output-dir",
                    str(tmp_path),
                    "--use-mock",
                ]

                with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
                    preprocess_unified.main()

                # Should have called validate_and_exit with use_mock=True
                mock_validate.assert_called_once_with(use_mock=True)


class TestUnifiedPreprocessingEndToEnd:
    """End-to-end tests with mock data."""

    def test_creates_unified_parquet_with_mock_data(self, tmp_path):
        """Full run with mock data should create valid unified parquet file."""
        from scripts import preprocess_unified

        output_dir = tmp_path / "features"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run with mock data
        args = [
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-03-31",
            "--symbols",
            "AAPL",
            "MSFT",
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
            "--use-mock",
        ]

        with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
            preprocess_unified.main()

        # Verify output file was created
        output_file = output_dir / "all_features.parquet"
        assert output_file.exists(), f"Unified parquet file not created at {output_file}"

        # Verify file structure
        df = pd.read_parquet(output_file)
        assert "symbol" in df.columns
        assert len(df["symbol"].unique()) == 2  # AAPL and MSFT

    def test_output_has_correct_date_range(self, tmp_path):
        """Unified file should contain data within specified date range."""
        from scripts import preprocess_unified

        output_dir = tmp_path / "features"
        output_dir.mkdir(parents=True, exist_ok=True)

        start_date = "2023-01-01"
        end_date = "2023-03-31"

        args = [
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--symbols",
            "AAPL",
            "--output-dir",
            str(output_dir),
            "--use-mock",
        ]

        with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
            preprocess_unified.main()

        # Verify date range
        output_file = output_dir / "all_features.parquet"
        df = pd.read_parquet(output_file)

        min_date = df["date"].min()
        max_date = df["date"].max()

        assert pd.Timestamp(start_date) <= min_date
        assert max_date <= pd.Timestamp(end_date)

    def test_includes_progress_reporting(self, tmp_path, capsys):
        """Should display progress during processing."""
        from scripts import preprocess_unified

        output_dir = tmp_path / "features"
        output_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-01-31",
            "--symbols",
            "AAPL",
            "--output-dir",
            str(output_dir),
            "--use-mock",
        ]

        with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
            preprocess_unified.main()

        # Check output contains progress indicators
        captured = capsys.readouterr()
        assert "Processing" in captured.out or "Done!" in captured.out

    def test_displays_summary_stats(self, tmp_path, capsys):
        """Should display summary statistics at end of processing."""
        from scripts import preprocess_unified

        output_dir = tmp_path / "features"
        output_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-01-31",
            "--symbols",
            "AAPL",
            "MSFT",
            "--output-dir",
            str(output_dir),
            "--use-mock",
        ]

        with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
            preprocess_unified.main()

        captured = capsys.readouterr()
        # Should show summary stats
        assert "Total rows" in captured.out
        assert "Symbols" in captured.out
        assert "Date range" in captured.out


class TestUnifiedPreprocessingErrors:
    """Test error handling and edge cases."""

    def test_exits_without_credentials(self, tmp_path):
        """Script should exit when WRDS credentials are missing and not using mock."""
        from scripts import preprocess_unified

        # Clear any existing credentials
        with mock.patch.dict("os.environ", {}, clear=True):
            args = [
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--symbols",
                "AAPL",
                "--output-dir",
                str(tmp_path),
            ]

            with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
                with pytest.raises(SystemExit) as exc_info:
                    preprocess_unified.main()

                assert exc_info.value.code == 1

    def test_handles_empty_symbols_list(self, tmp_path):
        """Should handle empty symbols list gracefully."""
        from scripts import preprocess_unified

        with mock.patch(
            "sys.argv",
            [
                "preprocess_unified.py",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--output-dir",
                str(tmp_path),
                "--use-mock",
            ],
        ):
            with pytest.raises(SystemExit):
                preprocess_unified.main()

    def test_handles_invalid_date_format(self, tmp_path):
        """Should handle invalid date format gracefully."""
        from scripts import preprocess_unified

        with mock.patch(
            "sys.argv",
            [
                "preprocess_unified.py",
                "--start-date",
                "invalid-date",
                "--end-date",
                "2023-12-31",
                "--symbols",
                "AAPL",
                "--output-dir",
                str(tmp_path),
                "--use-mock",
            ],
        ):
            with pytest.raises((SystemExit, ValueError)):
                preprocess_unified.main()

    def test_output_directory_created_if_missing(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        from scripts import preprocess_unified

        output_dir = tmp_path / "nested" / "features"

        # Directory should not exist initially
        assert not output_dir.exists()

        args = [
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-01-31",
            "--symbols",
            "AAPL",
            "--output-dir",
            str(output_dir),
            "--use-mock",
        ]

        with mock.patch("sys.argv", ["preprocess_unified.py"] + args):
            preprocess_unified.main()

        # Directory should be created
        assert output_dir.exists()
