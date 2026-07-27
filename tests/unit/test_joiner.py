from pathlib import Path

import pandas as pd
import pytest

from feature_generator.config import DatasetConfig, TableConfig
from feature_generator.dataset.joiner import join_tables

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TRANSACTION_PATH = FIXTURES / "mini_ieee_transaction.csv"
IDENTITY_PATH = FIXTURES / "mini_ieee_identity.csv"


def _config() -> DatasetConfig:
    return DatasetConfig(
        name="mini_ieee",
        tables=[
            TableConfig(name="transaction", path=str(TRANSACTION_PATH), role="primary"),
            TableConfig(name="identity", path=str(IDENTITY_PATH), role="secondary", join_key="TransactionID"),
        ],
        target_column="isFraud",
        target_table="transaction",
        id_columns=["TransactionID"],
        time_column="TransactionDT",
    )


def test_join_preserves_every_primary_row() -> None:
    transaction = pd.read_csv(TRANSACTION_PATH)
    joined = join_tables(_config())
    assert len(joined) == len(transaction)


def test_join_is_left_join_unmatched_rows_get_nan_identity_columns() -> None:
    identity = pd.read_csv(IDENTITY_PATH)
    joined = join_tables(_config())

    matched_ids = set(identity["TransactionID"])
    unmatched = joined[~joined["TransactionID"].isin(matched_ids)]
    matched = joined[joined["TransactionID"].isin(matched_ids)]

    assert len(unmatched) > 0  # fixture deliberately has partial identity coverage
    assert unmatched["DeviceType"].isna().all()
    assert matched["DeviceType"].notna().all()


def test_join_renames_colliding_non_key_columns_with_table_prefix() -> None:
    joined = join_tables(_config())
    # both tables have a "ProductCD" column (deliberate collision in the fixture)
    assert "ProductCD" in joined.columns  # from the primary table, untouched
    assert "identity_ProductCD" in joined.columns  # from the secondary table, renamed
    assert "TransactionID" in joined.columns  # the join key itself is never renamed


def test_join_target_and_time_columns_survive() -> None:
    joined = join_tables(_config())
    assert "isFraud" in joined.columns
    assert "TransactionDT" in joined.columns
    assert joined["TransactionDT"].is_monotonic_increasing  # fixture was generated pre-sorted


def test_join_raises_on_missing_join_key_config() -> None:
    config = _config()
    config.tables[1].join_key = None
    with pytest.raises(ValueError, match="no join_key configured"):
        join_tables(config)


def test_join_raises_when_join_key_absent_from_primary_table() -> None:
    config = _config()
    config.tables[1].join_key = "NotAColumn"
    with pytest.raises(ValueError, match="is not a column of the primary table"):
        join_tables(config)


def test_join_raises_when_no_primary_table_configured() -> None:
    config = _config()
    config.tables[0].role = "secondary"
    config.tables[0].join_key = "TransactionID"
    with pytest.raises(ValueError, match="no primary table"):
        join_tables(config)


def test_join_single_table_dataset_is_a_noop() -> None:
    config = DatasetConfig(
        name="single",
        tables=[TableConfig(name="transaction", path=str(TRANSACTION_PATH), role="primary")],
        target_column="isFraud",
        target_table="transaction",
    )
    joined = join_tables(config)
    original = pd.read_csv(TRANSACTION_PATH)
    pd.testing.assert_frame_equal(joined, original)
