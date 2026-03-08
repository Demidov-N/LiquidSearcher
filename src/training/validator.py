"""Cross-regime validation with embargo/purge periods."""

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationFold:
    """Single validation fold with metadata."""

    name: str
    start: str
    end: str
    regime: str
    embargo_before: str
    embargo_after: str


class CrossRegimeValidator:
    """Manages temporal data splits with purge and embargo periods.

    Implements the exact split structure:
    - Training: 2010-2018 (with 252-day purge from end)
    - Three validation folds: 2020, 2021-2022, 2023
    - Embargo gaps: 252 trading days (~1 year)
    - Test: 2024 (never touch until final)

    Attributes:
        train_start: Start of training period
        train_end: Effective end of training (after purge)
        val_folds: List of ValidationFold with regime descriptions
        test_start: Start of test period (final evaluation)
        test_end: End of test period
    """

    # Date constants
    TRAIN_START = "2010-01-01"
    RAW_TRAIN_END = "2019-12-31"
    PURGE_DAYS = 252
    EFFECTIVE_TRAIN_END = "2018-12-31"

    EMBARGO_1_START = "2019-01-01"
    EMBARGO_1_END = "2019-12-31"

    VAL_FOLD_1_START = "2020-01-01"
    VAL_FOLD_1_END = "2020-12-31"

    EMBARGO_2_START = "2021-01-01"
    EMBARGO_2_END = "2021-03-31"

    VAL_FOLD_2_START = "2021-04-01"
    VAL_FOLD_2_END = "2022-12-31"

    EMBARGO_3_START = "2023-01-01"
    EMBARGO_3_END = "2023-03-31"

    VAL_FOLD_3_START = "2023-04-01"
    VAL_FOLD_3_END = "2023-12-31"

    EMBARGO_4_START = "2024-01-01"
    EMBARGO_4_END = "2024-01-31"

    TEST_START = "2024-02-01"
    TEST_END = "2024-12-31"

    def __init__(self) -> None:
        """Initialize validator with all date ranges."""
        self.train_start = self.TRAIN_START
        self.train_end = self.EFFECTIVE_TRAIN_END

        # Define validation folds with regime descriptions
        self.val_folds = [
            ValidationFold(
                name="COVID Crash + Recovery",
                start=self.VAL_FOLD_1_START,
                end=self.VAL_FOLD_1_END,
                regime="High volatility, market crash, recovery",
                embargo_before=self.EMBARGO_1_START,
                embargo_after=self.EMBARGO_2_START,
            ),
            ValidationFold(
                name="Meme Stocks + Rate Shock",
                start=self.VAL_FOLD_2_START,
                end=self.VAL_FOLD_2_END,
                regime="Retail trading boom, inflation, rate hikes",
                embargo_before=self.EMBARGO_2_START,
                embargo_after=self.EMBARGO_3_START,
            ),
            ValidationFold(
                name="AI Boom / Soft Landing",
                start=self.VAL_FOLD_3_START,
                end=self.VAL_FOLD_3_END,
                regime="AI enthusiasm, soft landing narrative",
                embargo_before=self.EMBARGO_3_START,
                embargo_after=self.EMBARGO_4_START,
            ),
        ]

        self.test_start = self.TEST_START
        self.test_end = self.TEST_END

    def get_train_range(self) -> tuple[str, str]:
        """Get training date range (after purge)."""
        return (self.train_start, self.train_end)

    def get_val_fold(self, fold_idx: int) -> ValidationFold:
        """Get specific validation fold."""
        if fold_idx < 0 or fold_idx >= len(self.val_folds):
            raise ValueError(f"fold_idx must be 0, 1, or 2, got {fold_idx}")
        return self.val_folds[fold_idx]

    def iter_val_folds(self) -> Iterator[ValidationFold]:
        """Iterate over all validation folds."""
        yield from self.val_folds

    def get_test_range(self) -> tuple[str, str]:
        """Get test date range (final evaluation)."""
        return (self.test_start, self.test_end)

    def print_summary(self) -> None:
        """Print human-readable summary of data splits."""
        print("=" * 70)
        print("CROSS-REGIME VALIDATION STRUCTURE")
        print("=" * 70)

        print("\nTRAINING (clean, after purge)")
        print(f"  {self.train_start} → {self.train_end}")
        print(f"  (Purged last {self.PURGE_DAYS} days to prevent look-ahead)")

        print("\nEMBARGO GAP 1 (no data)")
        print(f"  {self.EMBARGO_1_START} → {self.EMBARGO_1_END}")
        print("  (252 trading days autocorrelation decay)")

        for i, fold in enumerate(self.val_folds):
            print(f"\nVALIDATION FOLD {i + 1}: {fold.name}")
            print(f"  {fold.start} → {fold.end}")
            print(f"  Regime: {fold.regime}")

        print(f"\n{'=' * 70}")
        print("FINAL TEST SET — NEVER TOUCH UNTIL COMPLETELY DONE")
        print(f"  {self.test_start} → {self.test_end}")
        print("=" * 70)

    def validate_no_overlap(self) -> bool:
        """Verify no data leakage between splits."""
        train_end = pd.Timestamp(self.train_end)

        for fold in self.val_folds:
            fold_start = pd.Timestamp(fold.start)

            if fold_start <= train_end:
                raise ValueError(
                    f"Data leakage: fold '{fold.name}' starts {fold_start} "
                    f"but training ends {train_end}"
                )

        test_start = pd.Timestamp(self.test_start)
        last_fold_end = pd.Timestamp(self.val_folds[-1].end)

        if test_start <= last_fold_end:
            raise ValueError(
                f"Data leakage: test starts {test_start} but last validation ends {last_fold_end}"
            )

        return True
