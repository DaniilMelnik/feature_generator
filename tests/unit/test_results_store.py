from pathlib import Path

import pytest

from feature_generator.knowledge_base.results_store import ResultsStore
from feature_generator.schemas import FeatureSpec, StaticCheckResult, ValidationResult


def _spec(name: str = "avg_room_service") -> FeatureSpec:
    return FeatureSpec(
        hypothesis_id="h1",
        feature_name=name,
        module_source="FEATURE_NAME = 'avg_room_service'\n",
        declared_input_columns=["HomePlanet", "RoomService"],
        declared_input_tables=["passengers"],
        output_dtype="float64",
        codegen_model="claude-sonnet-5",
    )


def test_promote_and_retrieve_feature(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results.duckdb")
    spec = _spec()
    result = ValidationResult(
        feature_name=spec.feature_name, hypothesis_id="h1",
        static=StaticCheckResult(passed=True), final_status="validated",
    )

    store.promote("run-1", spec, result)

    assert store.is_promoted("avg_room_service")
    assert store.get_feature_code("avg_room_service") == spec.module_source
    assert store.get_feature_spec("avg_room_service") == spec
    store.close()


def test_promote_rejects_non_validated_features(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results.duckdb")
    spec = _spec()
    result = ValidationResult(
        feature_name=spec.feature_name, hypothesis_id="h1",
        static=StaticCheckResult(passed=False, violations=["denied import"]),
        final_status="static_failed",
    )

    with pytest.raises(ValueError, match="refusing to promote"):
        store.promote("run-1", spec, result)

    assert not store.is_promoted(spec.feature_name)
    store.close()


def test_is_promoted_false_for_unknown_feature(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results.duckdb")
    assert store.is_promoted("does_not_exist") is False
    assert store.get_feature_code("does_not_exist") is None
    store.close()


def test_list_promoted_features_scoped_by_run(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results.duckdb")
    result = ValidationResult(
        feature_name="f1", hypothesis_id="h1", static=StaticCheckResult(passed=True),
        final_status="validated",
    )
    store.promote("run-1", _spec("f1"), result.model_copy(update={"feature_name": "f1"}))
    store.promote(
        "run-2",
        _spec("f2"),
        result.model_copy(update={"feature_name": "f2"}),
    )

    assert [r["feature_name"] for r in store.list_promoted_features("run-1")] == ["f1"]
    assert len(store.list_promoted_features()) == 2
    store.close()


def test_promote_stores_iteration_and_lift_retrievable_via_detail(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results.duckdb")
    spec = _spec()
    result = ValidationResult(
        feature_name=spec.feature_name, hypothesis_id="h1",
        static=StaticCheckResult(passed=True), final_status="validated",
    )

    store.promote("run-1", spec, result, iteration=2, lift_over_baseline=0.0135)

    detail = store.get_promoted_feature_detail(spec.feature_name)
    assert detail["iteration"] == 2
    assert detail["lift_over_baseline"] == pytest.approx(0.0135)
    assert detail["feature_spec"] == spec
    assert detail["validation_result"] == result
    store.close()


def test_promote_defaults_iteration_and_lift_when_not_provided(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results.duckdb")
    spec = _spec()
    result = ValidationResult(
        feature_name=spec.feature_name, hypothesis_id="h1",
        static=StaticCheckResult(passed=True), final_status="validated",
    )

    store.promote("run-1", spec, result)

    detail = store.get_promoted_feature_detail(spec.feature_name)
    assert detail["iteration"] == 0
    assert detail["lift_over_baseline"] is None
    store.close()


def test_get_promoted_feature_detail_returns_none_for_unknown_feature(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results.duckdb")
    assert store.get_promoted_feature_detail("does_not_exist") is None
    store.close()


def test_knowledge_base_and_results_store_are_separate_databases(tmp_path: Path) -> None:
    """A feature that fails validation must be absent from the results store
    even if it was fully recorded in the knowledge base -- the two stores are
    intentionally independent (see knowledge_base.db.KnowledgeBase).
    """
    from feature_generator.knowledge_base.db import KnowledgeBase

    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    store = ResultsStore(tmp_path / "results.duckdb")

    rejected = ValidationResult(
        feature_name="rejected_feature", hypothesis_id="h1",
        static=StaticCheckResult(passed=False, violations=["leak"]), final_status="leakage_rejected",
    )
    kb.add_validation_result("run-1", rejected)

    assert kb.list_validation_results("run-1")[0].feature_name == "rejected_feature"
    assert not store.is_promoted("rejected_feature")

    kb.close()
    store.close()
