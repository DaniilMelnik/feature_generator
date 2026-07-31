"""Append-only DuckDB knowledge base: every hypothesis, generated feature
spec, validation result, training run, and stability report a pipeline run
has ever produced -- including rejected/failed ones. This is what the
hypothesis agent's `kb_excerpt` is rendered from each iteration, so it can
avoid retrying failed ideas and build on validated ones.

Each table stores a handful of indexed/filterable columns plus the full
pydantic record as a JSON string, rather than a fully normalized relational
schema -- simple, robust to the schemas evolving, and sufficient for the
query patterns this pipeline actually needs (by run, by status, by id).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from feature_generator.schemas import (
    FeatureHypothesis,
    FeatureSpec,
    StabilityReport,
    TrainingMetrics,
    ValidationResult,
)

if TYPE_CHECKING:
    from feature_generator.modeling.feature_selection import FeatureSelectionOutput

_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS hypotheses (
        id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        iteration INTEGER NOT NULL,
        feature_type VARCHAR NOT NULL,
        proposed_by_model VARCHAR NOT NULL,
        data VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_specs (
        hypothesis_id VARCHAR NOT NULL,
        codegen_attempt INTEGER NOT NULL,
        feature_name VARCHAR NOT NULL,
        codegen_model VARCHAR NOT NULL,
        data VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (hypothesis_id, codegen_attempt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_results (
        run_id VARCHAR NOT NULL,
        feature_name VARCHAR NOT NULL,
        hypothesis_id VARCHAR NOT NULL,
        final_status VARCHAR NOT NULL,
        data VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (run_id, feature_name, hypothesis_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_metrics (
        run_id VARCHAR NOT NULL,
        iteration INTEGER NOT NULL,
        feature_set_id VARCHAR NOT NULL,
        auc_mean DOUBLE,
        data VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (run_id, iteration)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stability_reports (
        run_id VARCHAR NOT NULL,
        iteration INTEGER NOT NULL,
        feature_set_id VARCHAR NOT NULL,
        overall_stability_flag VARCHAR NOT NULL,
        data VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (run_id, iteration)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_selection_rounds (
        run_id VARCHAR NOT NULL,
        iteration INTEGER NOT NULL,
        baseline_auc DOUBLE,
        final_auc DOUBLE,
        data VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (run_id, iteration)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_metadata (
        run_id VARCHAR PRIMARY KEY,
        dataset_name VARCHAR NOT NULL,
        baseline_auc DOUBLE,
        raw_feature_columns VARCHAR NOT NULL,
        llm_backend VARCHAR,
        started_at TIMESTAMP DEFAULT current_timestamp
    )
    """,
]


class KnowledgeBase:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(self.db_path)
        for statement in _SCHEMA_STATEMENTS:
            self._conn.execute(statement)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "KnowledgeBase":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes --------------------------------------------------------

    def add_hypothesis(self, run_id: str, hypothesis: FeatureHypothesis) -> None:
        self._conn.execute(
            "INSERT INTO hypotheses (id, run_id, iteration, feature_type, proposed_by_model, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                hypothesis.id,
                run_id,
                hypothesis.iteration,
                hypothesis.feature_type,
                hypothesis.proposed_by_model,
                hypothesis.model_dump_json(),
            ],
        )

    def add_feature_spec(self, spec: FeatureSpec) -> None:
        self._conn.execute(
            "INSERT INTO feature_specs "
            "(hypothesis_id, codegen_attempt, feature_name, codegen_model, data) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                spec.hypothesis_id,
                spec.codegen_attempt,
                spec.feature_name,
                spec.codegen_model,
                spec.model_dump_json(),
            ],
        )

    def add_validation_result(self, run_id: str, result: ValidationResult) -> None:
        self._conn.execute(
            "INSERT INTO validation_results "
            "(run_id, feature_name, hypothesis_id, final_status, data) VALUES (?, ?, ?, ?, ?)",
            [
                run_id,
                result.feature_name,
                result.hypothesis_id,
                result.final_status,
                result.model_dump_json(),
            ],
        )

    def add_training_metrics(self, run_id: str, metrics: TrainingMetrics) -> None:
        # `feature_set_id` is content-derived (sorted feature names) and can
        # legitimately repeat across iterations (e.g. an iteration where no
        # new candidate survives leaves the accepted set unchanged) -- the
        # primary key is `(run_id, iteration)`, which is always unique, and
        # INSERT OR REPLACE makes this safe to call again for the same
        # iteration (e.g. a checkpoint-resume retry) without raising.
        self._conn.execute(
            "INSERT OR REPLACE INTO training_metrics (run_id, iteration, feature_set_id, auc_mean, data) "
            "VALUES (?, ?, ?, ?, ?)",
            [run_id, metrics.iteration, metrics.feature_set_id, metrics.auc_mean, metrics.model_dump_json()],
        )

    def add_stability_report(self, run_id: str, report: StabilityReport) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO stability_reports "
            "(run_id, iteration, feature_set_id, overall_stability_flag, data) VALUES (?, ?, ?, ?, ?)",
            [
                run_id,
                report.iteration,
                report.feature_set_id,
                report.overall_stability_flag,
                report.model_dump_json(),
            ],
        )

    def add_feature_selection_round(self, run_id: str, iteration: int, output: "FeatureSelectionOutput") -> None:
        self._conn.execute(
            "INSERT INTO feature_selection_rounds (run_id, iteration, baseline_auc, final_auc, data) "
            "VALUES (?, ?, ?, ?, ?)",
            [run_id, iteration, output.baseline_auc, output.final_auc, json.dumps(dataclasses.asdict(output))],
        )

    def set_run_metadata(
        self,
        run_id: str,
        dataset_name: str,
        baseline_auc: float | None,
        raw_feature_columns: list[str],
        llm_backend: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_metadata "
            "(run_id, dataset_name, baseline_auc, raw_feature_columns, llm_backend) "
            "VALUES (?, ?, ?, ?, ?)",
            [run_id, dataset_name, baseline_auc, json.dumps(raw_feature_columns), llm_backend],
        )

    # --- reads -----------------------------------------------------------

    def get_hypothesis(self, hypothesis_id: str) -> FeatureHypothesis | None:
        row = self._conn.execute(
            "SELECT data FROM hypotheses WHERE id = ?", [hypothesis_id]
        ).fetchone()
        return FeatureHypothesis.model_validate_json(row[0]) if row else None

    def list_hypotheses(self, run_id: str) -> list[FeatureHypothesis]:
        rows = self._conn.execute(
            "SELECT data FROM hypotheses WHERE run_id = ? ORDER BY iteration, created_at", [run_id]
        ).fetchall()
        return [FeatureHypothesis.model_validate_json(r[0]) for r in rows]

    def get_feature_specs(self, hypothesis_id: str) -> list[FeatureSpec]:
        rows = self._conn.execute(
            "SELECT data FROM feature_specs WHERE hypothesis_id = ? ORDER BY codegen_attempt",
            [hypothesis_id],
        ).fetchall()
        return [FeatureSpec.model_validate_json(r[0]) for r in rows]

    def list_validation_results(
        self, run_id: str, final_status: str | None = None
    ) -> list[ValidationResult]:
        if final_status is None:
            rows = self._conn.execute(
                "SELECT data FROM validation_results WHERE run_id = ? ORDER BY created_at", [run_id]
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM validation_results WHERE run_id = ? AND final_status = ? "
                "ORDER BY created_at",
                [run_id, final_status],
            ).fetchall()
        return [ValidationResult.model_validate_json(r[0]) for r in rows]

    def list_training_metrics(self, run_id: str) -> list[TrainingMetrics]:
        rows = self._conn.execute(
            "SELECT data FROM training_metrics WHERE run_id = ? ORDER BY created_at", [run_id]
        ).fetchall()
        return [TrainingMetrics.model_validate_json(r[0]) for r in rows]

    def get_latest_training_metrics(self, run_id: str) -> TrainingMetrics | None:
        row = self._conn.execute(
            "SELECT data FROM training_metrics WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            [run_id],
        ).fetchone()
        return TrainingMetrics.model_validate_json(row[0]) if row else None

    def get_best_training_metrics(self, run_id: str) -> TrainingMetrics | None:
        row = self._conn.execute(
            "SELECT data FROM training_metrics WHERE run_id = ? ORDER BY auc_mean DESC LIMIT 1",
            [run_id],
        ).fetchone()
        return TrainingMetrics.model_validate_json(row[0]) if row else None

    def list_feature_selection_rounds(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT iteration, baseline_auc, final_auc, data FROM feature_selection_rounds "
            "WHERE run_id = ? ORDER BY iteration",
            [run_id],
        ).fetchall()
        rounds = []
        for iteration, baseline_auc, final_auc, data in rows:
            round_dict = json.loads(data)
            round_dict.update(iteration=iteration, baseline_auc=baseline_auc, final_auc=final_auc)
            rounds.append(round_dict)
        return rounds

    def get_run_metadata(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT dataset_name, baseline_auc, raw_feature_columns, llm_backend, started_at "
            "FROM run_metadata WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        dataset_name, baseline_auc, raw_feature_columns, llm_backend, started_at = row
        return {
            "dataset_name": dataset_name,
            "baseline_auc": baseline_auc,
            "raw_feature_columns": json.loads(raw_feature_columns),
            "llm_backend": llm_backend,
            "started_at": started_at,
        }

    def render_kb_excerpt(self, run_id: str, max_chars: int = 8000) -> str:
        """A compact, deterministic text summary of everything tried so far
        in this run -- fed to the hypothesis agent each iteration so it can
        avoid repeating failed ideas and build on validated ones.
        """
        hypotheses = {h.id: h for h in self.list_hypotheses(run_id)}
        results = self.list_validation_results(run_id)

        if not hypotheses:
            return "No feature hypotheses have been tried yet."

        lines: list[str] = []
        results_by_hypothesis: dict[str, ValidationResult] = {r.hypothesis_id: r for r in results}
        for hypothesis in hypotheses.values():
            result = results_by_hypothesis.get(hypothesis.id)
            status = result.final_status if result else "pending"
            line = f"- [iter {hypothesis.iteration}] {hypothesis.description} -> status={status}"
            if result and result.dynamic and result.dynamic.single_feature_auc is not None:
                line += f", single_feature_auc={result.dynamic.single_feature_auc:.3f}"
            if result and result.leakage_review:
                line += f", leakage_review={result.leakage_review.verdict}"
            lines.append(line)

        excerpt = "\n".join(lines)
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 20] + "\n... (truncated)"
        return excerpt
