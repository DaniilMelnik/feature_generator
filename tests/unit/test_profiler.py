from pathlib import Path

import pytest

from feature_generator.config import DatasetConfig, TableConfig
from feature_generator.profiling.profiler import profile_tables
from feature_generator.profiling.schemas import DataProfile

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_spaceship_titanic.csv"


def _config() -> DatasetConfig:
    return DatasetConfig(
        name="mini_spaceship_titanic",
        tables=[TableConfig(name="passengers", path=str(FIXTURE_PATH), role="primary")],
        target_column="Transported",
        target_table="passengers",
        id_columns=["PassengerId"],
        time_column=None,
        known_leakage_columns=["Name"],
    )


def test_profile_tables_produces_valid_data_profile() -> None:
    profile = profile_tables(_config())
    assert isinstance(profile, DataProfile)
    assert profile.dataset_name == "mini_spaceship_titanic"
    assert 0.0 < profile.positive_rate < 1.0

    table = profile.primary_table()
    assert table.n_rows == 60
    assert table.n_cols == 14


def test_column_roles_assigned_correctly() -> None:
    profile = profile_tables(_config())
    by_name = {c.name: c for c in profile.primary_table().columns}

    assert by_name["Transported"].role == "target"
    assert by_name["PassengerId"].role == "id"
    assert by_name["Name"].role == "ignore"  # known_leakage_columns
    assert by_name["Age"].role == "feature"
    assert by_name["CryoSleep"].role == "feature"


def test_correlation_only_computed_for_feature_columns() -> None:
    profile = profile_tables(_config())
    by_name = {c.name: c for c in profile.primary_table().columns}

    # target/id/ignore columns must never carry a correlation value
    assert by_name["Transported"].correlation_with_target is None
    assert by_name["PassengerId"].correlation_with_target is None
    assert by_name["Name"].correlation_with_target is None

    # CryoSleep was constructed to be strongly correlated with the target
    cryo_corr = by_name["CryoSleep"].correlation_with_target
    assert cryo_corr is not None
    assert abs(cryo_corr) > 0.2


def test_numeric_vs_categorical_stats_routed_correctly() -> None:
    profile = profile_tables(_config())
    by_name = {c.name: c for c in profile.primary_table().columns}

    age = by_name["Age"]
    assert age.numeric_stats is not None
    assert age.categorical_stats is None
    assert 0 <= age.numeric_stats.min <= age.numeric_stats.max <= 79

    home_planet = by_name["HomePlanet"]
    assert home_planet.categorical_stats is not None
    assert home_planet.numeric_stats is None
    assert home_planet.categorical_stats.cardinality <= 3

    # object-dtype boolean-like column (has NaNs, so not pure numpy bool) must
    # still be routed to categorical stats, not numeric
    cryo = by_name["CryoSleep"]
    assert cryo.categorical_stats is not None
    assert cryo.numeric_stats is None


def test_missingness_and_constant_detection() -> None:
    profile = profile_tables(_config())
    by_name = {c.name: c for c in profile.primary_table().columns}

    age = by_name["Age"]
    assert age.n_missing == 3
    assert age.pct_missing == pytest.approx(3 / 60)
    assert age.is_constant is False


def test_profiling_notes_mention_known_leakage_columns() -> None:
    profile = profile_tables(_config())
    assert any("Name" in note for note in profile.profiling_notes)


def test_profile_tables_handles_ieee_shaped_multi_table_config_with_time_column() -> None:
    ieee_dir = FIXTURE_PATH.parent
    cfg = DatasetConfig(
        name="mini_ieee",
        tables=[
            TableConfig(name="transaction", path=str(ieee_dir / "mini_ieee_transaction.csv"), role="primary"),
            TableConfig(
                name="identity", path=str(ieee_dir / "mini_ieee_identity.csv"), role="secondary",
                join_key="TransactionID",
            ),
        ],
        target_column="isFraud",
        target_table="transaction",
        id_columns=["TransactionID"],
        time_column="TransactionDT",
    )

    profile = profile_tables(cfg)

    transaction_table = next(t for t in profile.tables if t.table_name == "transaction")
    time_col = next(c for c in transaction_table.columns if c.name == "TransactionDT")
    assert time_col.role == "time"
    assert time_col.numeric_stats is not None  # relative-time int column, not a datetime64 -- routed as numeric
    assert time_col.datetime_stats is None

    target_col = next(c for c in transaction_table.columns if c.name == "isFraud")
    assert target_col.role == "target"

    identity_table = next(t for t in profile.tables if t.table_name == "identity")
    assert identity_table.role == "secondary"
    assert all(c.correlation_with_target is None for c in identity_table.columns)


def test_secondary_table_correlation_is_null_and_noted() -> None:
    cfg = _config()
    cfg = cfg.model_copy(
        update={
            "tables": [
                *cfg.tables,
                TableConfig(
                    name="side",
                    path=str(FIXTURE_PATH),
                    role="secondary",
                    join_key="PassengerId",
                ),
            ]
        }
    )
    profile = profile_tables(cfg)
    side_table = next(t for t in profile.tables if t.table_name == "side")
    for col in side_table.columns:
        assert col.correlation_with_target is None
    assert any("side" in note and "join" in note.lower() for note in profile.profiling_notes)
