"""The results store: a separate DuckDB database holding ONLY promoted
("winning") features -- their code, lineage, and metrics -- distinct from
the full knowledge base of everything ever tried. This is the deliverable
artifact of a run: a small, reusable feature library.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

from feature_generator.schemas import FeatureSpec, StabilityReport, TrainingMetrics, ValidationResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS promoted_features (
    feature_name VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    hypothesis_id VARCHAR NOT NULL,
    module_source VARCHAR NOT NULL,
    output_dtype VARCHAR NOT NULL,
    auc_mean DOUBLE,
    stability_flag VARCHAR,
    iteration INTEGER,
    lift_over_baseline DOUBLE,
    feature_spec_json VARCHAR NOT NULL,
    validation_result_json VARCHAR NOT NULL,
    training_metrics_json VARCHAR,
    stability_report_json VARCHAR,
    promoted_at TIMESTAMP DEFAULT current_timestamp
)
"""


class ResultsStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(self.db_path)
        self._conn.execute(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResultsStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def promote(
        self,
        run_id: str,
        spec: FeatureSpec,
        validation_result: ValidationResult,
        training_metrics: TrainingMetrics | None = None,
        stability_report: StabilityReport | None = None,
        *,
        iteration: int = 0,
        lift_over_baseline: float | None = None,
    ) -> None:
        if validation_result.final_status not in ("validated", "promoted"):
            raise ValueError(
                f"refusing to promote feature '{spec.feature_name}' with "
                f"final_status='{validation_result.final_status}' -- only fully "
                "validated features may enter the results store"
            )
        self._conn.execute(
            """
            INSERT INTO promoted_features (
                feature_name, run_id, hypothesis_id, module_source, output_dtype,
                auc_mean, stability_flag, iteration, lift_over_baseline,
                feature_spec_json, validation_result_json,
                training_metrics_json, stability_report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                spec.feature_name,
                run_id,
                spec.hypothesis_id,
                spec.module_source,
                spec.output_dtype,
                training_metrics.auc_mean if training_metrics else None,
                stability_report.overall_stability_flag if stability_report else None,
                iteration,
                lift_over_baseline,
                spec.model_dump_json(),
                validation_result.model_dump_json(),
                training_metrics.model_dump_json() if training_metrics else None,
                stability_report.model_dump_json() if stability_report else None,
            ],
        )

    def is_promoted(self, feature_name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM promoted_features WHERE feature_name = ?", [feature_name]
        ).fetchone()
        return row is not None

    def get_feature_code(self, feature_name: str) -> str | None:
        row = self._conn.execute(
            "SELECT module_source FROM promoted_features WHERE feature_name = ?", [feature_name]
        ).fetchone()
        return row[0] if row else None

    def list_promoted_features(self, run_id: str | None = None) -> list[dict]:
        if run_id is None:
            rows = self._conn.execute(
                "SELECT feature_name, run_id, auc_mean, stability_flag, promoted_at "
                "FROM promoted_features ORDER BY promoted_at"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT feature_name, run_id, auc_mean, stability_flag, promoted_at "
                "FROM promoted_features WHERE run_id = ? ORDER BY promoted_at",
                [run_id],
            ).fetchall()
        columns = ["feature_name", "run_id", "auc_mean", "stability_flag", "promoted_at"]
        return [dict(zip(columns, row)) for row in rows]

    def get_feature_spec(self, feature_name: str) -> FeatureSpec | None:
        row = self._conn.execute(
            "SELECT feature_spec_json FROM promoted_features WHERE feature_name = ?", [feature_name]
        ).fetchone()
        return FeatureSpec.model_validate_json(row[0]) if row else None

    def get_promoted_feature_detail(self, feature_name: str) -> dict | None:
        """Every stored field for one promoted feature, JSON blobs parsed back
        into their pydantic models -- the full detail a report needs.
        """
        row = self._conn.execute(
            """
            SELECT feature_name, run_id, hypothesis_id, module_source, output_dtype,
                   auc_mean, stability_flag, iteration, lift_over_baseline,
                   feature_spec_json, validation_result_json,
                   training_metrics_json, stability_report_json, promoted_at
            FROM promoted_features WHERE feature_name = ?
            """,
            [feature_name],
        ).fetchone()
        if row is None:
            return None
        (
            feature_name,
            run_id,
            hypothesis_id,
            module_source,
            output_dtype,
            auc_mean,
            stability_flag,
            iteration,
            lift_over_baseline,
            feature_spec_json,
            validation_result_json,
            training_metrics_json,
            stability_report_json,
            promoted_at,
        ) = row
        return {
            "feature_name": feature_name,
            "run_id": run_id,
            "hypothesis_id": hypothesis_id,
            "module_source": module_source,
            "output_dtype": output_dtype,
            "auc_mean": auc_mean,
            "stability_flag": stability_flag,
            "iteration": iteration,
            "lift_over_baseline": lift_over_baseline,
            "feature_spec": FeatureSpec.model_validate_json(feature_spec_json),
            "validation_result": ValidationResult.model_validate_json(validation_result_json),
            "training_metrics": TrainingMetrics.model_validate_json(training_metrics_json)
            if training_metrics_json
            else None,
            "stability_report": StabilityReport.model_validate_json(stability_report_json)
            if stability_report_json
            else None,
            "promoted_at": promoted_at,
        }
