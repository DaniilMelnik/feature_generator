from pathlib import Path

from feature_generator.sandbox.static_checks import run_static_checks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_compliant_fixture_passes() -> None:
    source = (FIXTURES / "compliant_feature_groupby_mean.py").read_text()
    result = run_static_checks(
        source,
        forbidden_target_column="Transported",
        declared_input_columns=["HomePlanet", "RoomService"],
    )
    assert result.passed, result.violations
    assert result.violations == []
    assert result.ast_denylist_hits == []


def test_malicious_os_import_fixture_is_rejected() -> None:
    source = (FIXTURES / "malicious_feature_os_import.py").read_text()
    result = run_static_checks(
        source, forbidden_target_column="Transported", declared_input_columns=["Age"]
    )
    assert not result.passed
    assert "import:os" in result.ast_denylist_hits
    assert any("os" in v for v in result.violations)


def test_denied_call_eval_is_rejected() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["Age"]

def fit(train_df, context):
    return {"v": eval("1+1")}

def transform(df, params):
    return df["Age"]
"""
    result = run_static_checks(source, forbidden_target_column="Transported")
    assert not result.passed
    assert "call:eval" in result.ast_denylist_hits


def test_denied_call_open_is_rejected() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["Age"]

def fit(train_df, context):
    with open("/etc/passwd") as f:
        data = f.read()
    return {"v": data}

def transform(df, params):
    return df["Age"]
"""
    result = run_static_checks(source, forbidden_target_column="Transported")
    assert not result.passed
    assert "call:open" in result.ast_denylist_hits


def test_denied_dunder_attribute_sandbox_escape_is_rejected() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["Age"]

def fit(train_df, context):
    leak = ().__class__.__bases__[0].__subclasses__()
    return {"v": len(leak)}

def transform(df, params):
    return df["Age"]
"""
    result = run_static_checks(source, forbidden_target_column="Transported")
    assert not result.passed
    assert "attr:__subclasses__" in result.ast_denylist_hits
    assert "attr:__bases__" in result.ast_denylist_hits


def test_import_not_in_allowlist_is_rejected() -> None:
    source = """
import requests

FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["Age"]

def fit(train_df, context):
    return {}

def transform(df, params):
    return df["Age"]
"""
    result = run_static_checks(source, forbidden_target_column="Transported")
    assert not result.passed
    assert any("requests" in v for v in result.violations)


def test_allowlisted_imports_pass() -> None:
    source = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.stats

FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["Age"]

def fit(train_df, context):
    return {"mean": float(np.mean(train_df["Age"]))}

def transform(df, params):
    return df["Age"] - params["mean"]
"""
    result = run_static_checks(source, forbidden_target_column="Transported")
    assert result.passed, result.violations


def test_target_column_literal_reference_is_rejected() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["Age"]

def fit(train_df, context):
    return {"leak": train_df["Transported"].mean()}

def transform(df, params):
    return df["Age"]
"""
    result = run_static_checks(source, forbidden_target_column="Transported")
    assert not result.passed
    assert "target-column-literal" in result.ast_denylist_hits


def test_target_column_attribute_reference_is_rejected() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["Age"]

def fit(train_df, context):
    return {"leak": train_df.Transported.mean()}

def transform(df, params):
    return df["Age"]
"""
    result = run_static_checks(source, forbidden_target_column="Transported")
    assert not result.passed
    assert "target-column-attr" in result.ast_denylist_hits


def test_missing_required_columns_literal_is_rejected() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = compute_columns()

def fit(train_df, context):
    return {}

def transform(df, params):
    return df["Age"]
"""
    result = run_static_checks(source, forbidden_target_column="Transported")
    assert not result.passed
    assert any("REQUIRED_COLUMNS must be a literal" in v for v in result.violations)


def test_required_columns_mismatch_with_declared_is_rejected() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["Age", "VIP"]

def fit(train_df, context):
    return {}

def transform(df, params):
    return df["Age"]
"""
    result = run_static_checks(
        source, forbidden_target_column="Transported", declared_input_columns=["Age"]
    )
    assert not result.passed
    assert any("does not match" in v for v in result.violations)


def test_target_reference_in_fit_allowed_when_opted_in() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["HomePlanet"]

def fit(train_df, context):
    return {"means": train_df.groupby("HomePlanet")["Transported"].mean().to_dict()}

def transform(df, params):
    return df["HomePlanet"].map(params["means"])
"""
    result = run_static_checks(
        source, forbidden_target_column="Transported", allow_target_in_fit=True
    )
    assert result.passed, result.violations


def test_target_reference_in_transform_always_rejected_even_when_opted_in() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["HomePlanet"]

def fit(train_df, context):
    return {"means": train_df.groupby("HomePlanet")["Transported"].mean().to_dict()}

def transform(df, params):
    return df["Transported"]
"""
    result = run_static_checks(
        source, forbidden_target_column="Transported", allow_target_in_fit=True
    )
    assert not result.passed
    assert "target-column-literal" in result.ast_denylist_hits


def test_target_reference_in_fit_still_rejected_when_not_opted_in() -> None:
    source = """
FEATURE_NAME = "x"
REQUIRED_COLUMNS = ["HomePlanet"]

def fit(train_df, context):
    return {"means": train_df.groupby("HomePlanet")["Transported"].mean().to_dict()}

def transform(df, params):
    return df["HomePlanet"].map(params["means"])
"""
    result = run_static_checks(
        source, forbidden_target_column="Transported", allow_target_in_fit=False
    )
    assert not result.passed
    assert "target-column-literal" in result.ast_denylist_hits


def test_syntax_error_is_reported_as_violation() -> None:
    result = run_static_checks("def fit(:\n    pass", forbidden_target_column="Transported")
    assert not result.passed
    assert any("syntax error" in v for v in result.violations)
