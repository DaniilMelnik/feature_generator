from pathlib import Path

import pandas as pd
import pytest

from feature_generator.sandbox.contract import (
    ContractViolationError,
    FitContext,
    InProcessFeatureComputer,
    load_trusted_module_from_file,
    validate_module_structure,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HomePlanet": ["Earth", "Earth", "Mars", "Mars", "Europa"],
            "RoomService": [0.0, 50.0, 10.0, 30.0, 200.0],
            "Age": [22, 35, 41, 19, 60],
            "Transported": [True, False, True, False, True],
        }
    )


def test_load_and_wrap_compliant_fixture() -> None:
    module = load_trusted_module_from_file(FIXTURES / "compliant_feature_groupby_mean.py")
    assert validate_module_structure(module) == []

    computer = InProcessFeatureComputer(module)
    assert computer.feature_name == "avg_room_service_by_home_planet"
    assert computer.required_columns == ["HomePlanet", "RoomService"]

    df = _sample_df()
    context = FitContext(target_column="Transported")
    params = computer.fit(df, context)
    output = computer.transform(df, params)

    assert len(output) == len(df)
    # Earth mean RoomService = (0+50)/2 = 25
    assert output.iloc[0] == pytest.approx(25.0)


def test_in_process_computer_never_sees_columns_outside_required() -> None:
    """The dataset builder must never be able to leak extra columns (incl. the
    target) into a feature module -- InProcessFeatureComputer slices the
    dataframe to REQUIRED_COLUMNS before the module's fit/transform ever run,
    regardless of what else is present in the input.
    """

    seen_columns: list[list[str]] = []

    class _RecordingModule:
        FEATURE_NAME = "recorder"
        REQUIRED_COLUMNS = ["Age"]

        def fit(self, train_df, context):
            seen_columns.append(list(train_df.columns))
            return {}

        def transform(self, df, params):
            seen_columns.append(list(df.columns))
            return df["Age"]

    computer = InProcessFeatureComputer(_RecordingModule())
    df = _sample_df()  # has HomePlanet, RoomService, Age, Transported
    params = computer.fit(df, FitContext(target_column="Transported"))
    computer.transform(df, params)

    assert seen_columns == [["Age"], ["Age"]]
    assert all("Transported" not in cols for cols in seen_columns)


def test_validate_module_structure_reports_missing_attrs() -> None:
    class _Empty:
        pass

    violations = validate_module_structure(_Empty())
    assert "missing required attribute 'FEATURE_NAME'" in violations
    assert "missing required attribute 'REQUIRED_COLUMNS'" in violations
    assert "missing required attribute 'fit'" in violations
    assert "missing required attribute 'transform'" in violations


def test_validate_module_structure_reports_wrong_types() -> None:
    class _Bad:
        FEATURE_NAME = 123  # should be str
        REQUIRED_COLUMNS = "Age"  # should be list[str]
        fit = "not callable"
        transform = "also not callable"

    violations = validate_module_structure(_Bad())
    assert "FEATURE_NAME must be a str" in violations
    assert "REQUIRED_COLUMNS must be a list[str]" in violations
    assert "'fit' must be callable" in violations
    assert "'transform' must be callable" in violations


def test_allow_target_in_fit_exposes_target_to_fit_but_never_to_transform() -> None:
    seen_fit_columns: list[list[str]] = []
    seen_transform_columns: list[list[str]] = []

    class _TargetEncodingModule:
        FEATURE_NAME = "target_encoded"
        REQUIRED_COLUMNS = ["HomePlanet"]

        def fit(self, train_df, context):
            seen_fit_columns.append(list(train_df.columns))
            means = train_df.groupby("HomePlanet")["Transported"].mean()
            return {"means": means.to_dict()}

        def transform(self, df, params):
            seen_transform_columns.append(list(df.columns))
            return df["HomePlanet"].map(params["means"])

    computer = InProcessFeatureComputer(_TargetEncodingModule(), allow_target_in_fit=True)
    df = _sample_df()
    params = computer.fit(df, FitContext(target_column="Transported"))
    computer.transform(df, params)

    assert "Transported" in seen_fit_columns[0]
    assert "Transported" not in seen_transform_columns[0]


def test_allow_target_in_fit_defaults_to_false() -> None:
    class _RequiresTarget:
        FEATURE_NAME = "x"
        REQUIRED_COLUMNS = ["Age"]

        def fit(self, train_df, context):
            return {"v": train_df["Transported"].mean()}  # would KeyError without opt-in

        def transform(self, df, params):
            return df["Age"]

    computer = InProcessFeatureComputer(_RequiresTarget())
    with pytest.raises(KeyError):
        computer.fit(_sample_df(), FitContext(target_column="Transported"))


def test_in_process_computer_raises_contract_violation_for_malformed_module() -> None:
    class _Incomplete:
        FEATURE_NAME = "incomplete"

    with pytest.raises(ContractViolationError):
        InProcessFeatureComputer(_Incomplete())
